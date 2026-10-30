"""Local-only Nakhla course evaluation application.

NOT for deployment: course fixtures supply unverified employee emails. This app is
restricted to 127.0.0.1 and approval-based writes are opt-in with
EVAL_ENABLE_WRITES=1. Production authentication is in agent_backend.py.

Does not change the simulated kit under services/. The user must see the exact
server-side ticket draft and approve its action_id before an MCP write occurs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sys
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
WRITE_ENABLED = os.getenv("EVAL_ENABLE_WRITES") == "1"
app = FastAPI(title="Nakhla isolated course evaluator v3 — NOT FOR DEPLOYMENT")


class ChatUser(BaseModel):
    email: str | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    user: ChatUser | None = None


class ApproveRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    action_id: str = Field(min_length=8, max_length=128)
    approved: bool


@dataclass
class Session:
    email: str = ""
    last_error: str = ""
    last_category: str = ""
    turns: list[str] = field(default_factory=list)
    pending: dict | None = None
    consumed: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


sessions: dict[str, Session] = {}


@app.middleware("http")
async def loopback_only(request: Request, call_next):
    peer = request.client.host if request.client else ""
    try:
        allowed = ip_address(peer).is_loopback
    except ValueError:
        allowed = peer == "testclient"
    if not allowed:
        return JSONResponse(status_code=403, content={"reply": "Local evaluation only"})
    return await call_next(request)


def _mcp_text(result: Any) -> dict:
    if getattr(result, "isError", False):
        return {"ok": False, "error": {"code": "mcp_error", "message": "Tool call failed"}}
    for part in getattr(result, "content", []):
        if getattr(part, "type", "") == "text":
            try:
                value = json.loads(part.text)
                return value if isinstance(value, dict) else {"ok": False}
            except (ValueError, TypeError):
                pass
    return {"ok": False, "error": {"code": "invalid_mcp_response"}}


async def mcp_call(tool: str, args: dict | None = None) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "services.helpdesk_mcp"],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            return _mcp_text(await client.call_tool(tool, arguments=args or {}))


def _source(path: Path) -> dict | None:
    if path.suffix.lower() == ".md":
        raw = path.read_text(encoding="utf-8-sig")
        parts = raw.split("---", 2)
        if len(parts) != 3:
            return None
        meta = yaml.safe_load(parts[1]) or {}
        content = parts[2]
    elif path.suffix.lower() == ".pdf":
        import pymupdf
        with pymupdf.open(path) as pdf:
            content = "\n".join(page.get_text() for page in pdf)
        meta = {}
    else:
        return None
    if not isinstance(meta, dict):
        return None
    match = re.search(r"\b(KB|SP)-\d{3}\b", path.name.upper())
    doc_id = str(meta.get("id") or (match.group() if match else "")).upper()
    if not doc_id:
        return None
    # Never use deprecated, draft, or explicitly nonauthoritative documents.
    if str(meta.get("status", "current")).lower() != "current":
        return None
    if meta.get("authoritative") is False:
        return None
    return {"id": doc_id, "meta": meta, "text": content, "path": str(path)}


def load_docs() -> dict[str, dict]:
    kb = ROOT / "dataset" / "knowledge_base"
    docs: dict[str, dict] = {}
    if not kb.exists():
        logger.error("Knowledge base path missing: %s", kb)
        return docs
    for path in sorted(kb.rglob("*")):
        if path.suffix.lower() not in {".md", ".pdf"}:
            continue
        try:
            doc = _source(path)
            if doc:
                docs[doc["id"]] = doc
        except Exception:
            logger.exception("Could not load document %s", path)
    return docs


DOCS = load_docs()
CODE = re.compile(r"\b(?:NX|NID|SWC|CRM|HW)-\d{3,5}\b", re.I)


def doc_for_code(code: str) -> dict | None:
    # Prefer current, authoritative documents whose metadata or body covers
    # the exact code. Never substitute a deprecated predecessor (KB-001).
    found = []
    for doc in DOCS.values():
        meta_codes = [str(x).upper() for x in doc["meta"].get("error_codes", []) or []]
        if code.upper() in meta_codes or re.search(r"\b" + re.escape(code) + r"\b", doc["text"], re.I):
            preferred = doc["id"] == ("KB-002" if code.startswith("NX-") else "KB-006" if code.startswith("NID-") else "")
            found.append((int(preferred), int(code.upper() in meta_codes), doc))
    return sorted(found, key=lambda x: (x[0], x[1]), reverse=True)[0][2] if found else None


def _relevant_excerpt(doc: dict, code: str, max_lines: int = 8) -> str:
    text = doc["text"].strip()
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if re.search(r"\b" + re.escape(code) + r"\b", line, re.I)]
    if not matches:
        return ""

    # Some KBs (e.g. the NID-423 lockout guide) put the error code only
    # in the main title, with the actionable instructions in later sections.
    # A heading-only excerpt would hide the actual fix. Include the current
    # guide's body in those cases; never use draft/deprecated docs (load_docs).
    main_title = next((line for line in lines if line.startswith("# ")), "")
    if re.search(r"\b" + re.escape(code) + r"\b", main_title, re.I):
        return text[:5500]

    # For error-code tables, extract the relevant row's steps.
    rows = [line for line in lines if re.match(r"\s*\|\s*" + re.escape(code) + r"\s*\|", line, re.I)]
    if rows:
        pieces = [p.strip() for p in rows[0].strip().strip("|").split("|")]
        if len(pieces) >= 3 and pieces[2]:
            return pieces[2]

    idx = matches[-1] if len(matches) > 2 else matches[0]
    start = max(0, idx - 1)
    end = min(len(lines), idx + max_lines)
    section = []
    for line in lines[start:end]:
        if section and line.startswith("## "):
            break
        section.append(line)
    excerpt = "\n".join(section).strip()
    # If we only found a heading, move into the following sections.
    if len(excerpt) < 120:
        return "\n".join(lines[start:min(len(lines), start + 60)]).strip()[:5500]
    return excerpt[:3500]


def _citations(ids: list[str]) -> list[dict]:
    return [{"id": x} for x in dict.fromkeys(ids) if x in DOCS or x.startswith("INC-")]


def _reply(text: str, refs: list[str] | None = None, *, tools: list[dict] | None = None,
           pending: dict | None = None, result: dict | None = None, workflow: str = "answered") -> dict:
    data = {"reply": text, "citations": _citations(refs or []),
            "tool_calls": tools or [], "pending_action": pending, "state": workflow}
    if result is not None:  # The eval requires result to be absent in non-actions.
        data["result"] = result
    return data


def _draft(session: Session, subject: str, description: str, category: str, priority: str,
           employee: dict, kb_refs: list[str], attempted: list[str], tools: list[dict]) -> dict:
    if not isinstance(employee, dict) or not employee.get("email") or not employee.get("department"):
        return _reply("I need your work email to look you up before preparing a ticket.", tools=tools)
    if str(employee.get("account_status", "active")).lower() not in {"active", "enabled"}:
        return _reply("This account needs manual Service Desk verification before a ticket can be proposed.", tools=tools)
    payload = {
        "subject": subject[:180],
        "description": description[:4000],
        "category": category,
        "priority": priority,
        "requester_email": employee["email"],
        "department": employee["department"],
        "affected_users": 1,
        "attempted_steps": attempted or ["Reported the issue to the Service Desk; no troubleshooting steps confirmed"],
        "kb_refs": [x for x in dict.fromkeys(kb_refs) if x in DOCS],
    }
    action_id = secrets.token_urlsafe(24)
    pending = {
        "type": "create_ticket",
        "action_id": action_id,
        "summary": f"Create {priority} {category} ticket: {subject[:110]}",
        "payload": payload,
    }
    session.pending = pending
    return _reply(
        "I prepared a support ticket draft. Please review the exact details below and choose Approve or Reject. No ticket has been created yet.",
        payload["kb_refs"], tools=tools, pending=pending, workflow="awaiting_approval",
    )


def _already_tried(msg: str) -> bool:
    return bool(re.search(r"\b(already|tried|did all|still fail|still not|does not work|didn't work|updated|restarted|followed)\b", msg, re.I))


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "isolated_eval", "write_approval_enabled": WRITE_ENABLED,
            "loaded_docs": len(DOCS)}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    session = sessions.setdefault(body.session_id, Session())
    # A new chat turn invalidates any previous unapproved draft.
    session.pending = None
    msg = body.message.strip()
    code_match = CODE.search(msg)
    code = code_match.group(0).upper() if code_match else session.last_error
    email = (body.user.email or "").strip().lower() if body.user else session.email
    if session.email and email and email != session.email:
        # Do not carry conversation history across different directory identities.
        session.last_error, session.last_category = "", ""
        session.turns.clear()
    tools: list[dict] = []
    employee = None
    if email:
        try:
            found = await mcp_call("lookup_employee", {"email": email})
            tools.append({"tool": "lookup_employee", "ok": bool(found.get("ok"))})
            employee = found.get("employee") if found.get("ok") else None
            if isinstance(employee, dict) and employee.get("email", "").lower() == email:
                session.email = email
            else:
                employee = None
        except Exception:
            logger.exception("Directory lookup failed")
            tools.append({"tool": "lookup_employee", "ok": False})
    department = (employee or {}).get("department", "")
    lower = msg.lower()
    vpn = "vpn" in lower or "nakhlaconnect" in lower or code.startswith("NX-") or session.last_category == "vpn"
    if code_match:
        session.last_error = code
    session.last_category = "vpn" if vpn else "other"
    session.turns.append(msg)
    session.turns = session.turns[-10:]

    # An ambiguous request must be clarified, not escalated and not sourced.
    if vpn and not code and not email:
        return _reply("Please share your work email and the VPN error code (such as NX-112 or NX-503), and tell me what happens when you connect.", tools=tools, workflow="needs_clarification")
    if vpn and not code:
        return _reply("What exact VPN error code (NX-...) do you see? Please also describe the symptoms.", tools=tools, workflow="needs_clarification")

    # Check live incidents for VPN or infrastructure service disruptions.
    if vpn or "outage" in lower or "down" in lower:
        try:
            status = await mcp_call("get_service_status")
            tools.append({"tool": "get_service_status", "ok": bool(status.get("ok"))})
        except Exception:
            logger.exception("Service status lookup failed")
            status = {"ok": False}
            tools.append({"tool": "get_service_status", "ok": False})
        if not status.get("ok"):
            return _reply("I cannot verify the current service status. Please contact the Service Desk before changing VPN settings.", tools=tools, workflow="needs_verification")
        snapshot = status.get("status") or {}
        active = set(snapshot.get("active_incidents") or [])
        for incident in snapshot.get("incidents") or []:
            if incident.get("id") not in active or incident.get("status") != "active":
                continue
            codes = [str(x).upper() for x in incident.get("error_codes", [])]
            if code not in codes or not vpn:
                continue
            if incident.get("service") != "vpn":
                continue
            gw = (employee or {}).get("vpn_gateway") or ""
            cited = re.findall(r"\bGW-\d+\b", incident.get("affected") or "", re.I)
            if gw and gw.upper() not in {g.upper() for g in cited}:
                continue
            if not gw:
                return _reply("Which VPN gateway are you using (GW-1, GW-2, or GW-3)?", tools=tools, workflow="needs_clarification")
            return _reply(
                f"Active incident {incident['id']}: {incident.get('title', '')}.\n\n"
                f"Workaround: {incident.get('workaround', '')}\n\n"
                f"Instructions: {incident.get('instructions', '')}\n\nSources: {incident['id']}",
                [incident["id"]], tools=tools,
            )

    # Special Finance policy OVERRIDES generic VPN steps.
    finance = vpn and str(department).lower() == "finance"
    if finance:
        refs = [x for x in ("KB-016", "KB-002") if x in DOCS]
        if re.search(r"\b(ticket|escalat|security|need help|not working|fail|shows)\b", lower):
            return _draft(
                session,
                f"Finance VPN access: {code or 'connection failure'}",
                f"Finance user reports: {msg}. Follow Security Operations procedure; do not suggest generic VPN workarounds.",
                "security", "P2", employee, refs,
                ["Reported Finance VPN access failure; referred to Security Operations"], tools,
            )
        return _reply("Finance VPN must be handled by Security Operations. Do not switch gateways or use general VPN troubleshooting.", refs, tools=tools)

    # Unknown documented code: say so instead of inventing a fix.
    doc = doc_for_code(code) if code else None
    if code and not doc:
        return _reply(f"I could not find an approved knowledge-base fix for {code}. Please contact the Service Desk for escalation or ask to open a support ticket. I will not invent troubleshooting steps.", tools=tools)

    # Explicit ticket request, or after documented steps failed.
    explicit_ticket = bool(re.search(r"\b(open|create|raise|submit|file)\b.{0,40}\bticket\b|\bticket\b.{0,40}\b(open|create)\b", lower))
    follow_up_failure = bool(session.last_error and len(session.turns) >= 2 and _already_tried(msg))
    if explicit_ticket or follow_up_failure:
        refs = [doc["id"]] if doc else []
        if vpn:
            cat, priority = "network_access", "P2"
        elif "dock" in lower or "monitor" in lower or "hardware" in lower:
            cat, priority = "hardware", "P3"
        elif "software" in lower or "install" in lower or code.startswith("SWC-"):
            cat, priority = "software", "P3"
        else:
            cat, priority = "other", "P3"
        subject = f"{('VPN' if vpn else cat.title())} issue {code}".strip()
        attempted = ["User reports completing troubleshooting: " + msg] if _already_tried(msg) else ["Reported the issue; troubleshooting steps not confirmed"]
        return _draft(session, subject, "\n".join(session.turns), cat, priority,
                      employee, refs, attempted, tools)

    if doc:
        excerpt = _relevant_excerpt(doc, code)
        if not excerpt:
            return _reply("I found a knowledge-base reference but could not verify the exact procedure. Please contact the Service Desk.", tools=tools)
        # Provide an exact, current KB excerpt, not an LLM-generated workaround.
        return _reply(
            f"According to {doc['id']} (current knowledge base), for {code}:\n\n{excerpt}\n\n"
            "If these steps do not resolve the issue, tell me what you already tried so I can prepare a ticket for your approval."
            f"\n\nSources: {doc['id']}",
            [doc["id"]], tools=tools,
        )

    return _reply("I need a little more detail: which application, what error code or symptoms, and your work email? I will check the current approved knowledge base before recommending steps.", tools=tools, workflow="needs_clarification")




def _created_ticket_id(result: dict) -> str | None:
    """Unwrap the MCP success wrapper; the kit may return {'ticket': {...}}.

    Only IDs from an actual successful MCP response are accepted. Never infer
    an ID from the expected next ticket number or from a failed tool call.
    """
    if not isinstance(result, dict) or result.get("ok") is not True:
        return None
    candidates = [result]
    for key in ("ticket", "data", "result", "response", "created_ticket"):
        value = result.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for item in candidates:
        ticket_id = item.get("ticket_id")
        if isinstance(ticket_id, str) and re.fullmatch(r"IT-\d+", ticket_id):
            return ticket_id
    return None

@app.post("/api/approve")
async def approve(body: ApproveRequest):
    session = sessions.get(body.session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"reply": "No pending draft for this session.", "pending_action": None})
    async with session.lock:
        pending = session.pending
        if pending is None or pending.get("action_id") != body.action_id:
            return JSONResponse(status_code=409, content={"reply": "That draft is no longer pending or does not match this session.", "pending_action": None})
        session.pending = None  # One-time approval token is consumed immediately.
        session.consumed.add(body.action_id)
        if not body.approved:
            return _reply("Ticket request rejected. No ticket was created.", workflow="rejected")
        if not WRITE_ENABLED:
            return JSONResponse(status_code=403, content={
                "reply": "Ticket creation is disabled. Set EVAL_ENABLE_WRITES=1 only in the isolated course evaluation process.",
                "pending_action": None,
            })
        payload = pending["payload"]
        try:
            result = await mcp_call("create_ticket", {
                **payload, "idempotency_key": body.action_id,
            })
        except Exception:
            logger.exception("Failed to call create_ticket")
            result = {"ok": False, "error": {"code": "tool_unavailable"}}
        tool_calls = [{"tool": "create_ticket", "ok": bool(result.get("ok"))}]
        ticket_id = _created_ticket_id(result)
        if ticket_id:
            return _reply(f"Ticket created successfully: {ticket_id}", tools=tool_calls,
                          result={"ticket_id": ticket_id}, workflow="created")

        # Log just error metadata to distinguish API validation from an
        # unexpected MCP response shape; never fabricate a ticket ID.
        error_details = result.get("error") if isinstance(result, dict) else None
        code = (error_details or {}).get("code", "unexpected_mcp_response") if isinstance(error_details, dict) else "unexpected_mcp_response"
        logger.warning(
            "create_ticket not confirmed: ok=%s keys=%s error=%s",
            result.get("ok") if isinstance(result, dict) else None,
            sorted(result.keys()) if isinstance(result, dict) else [],
            {k: error_details.get(k) for k in ("code", "fields") if k in error_details}
            if isinstance(error_details, dict) else None,
        )
        return _reply(f"Ticket creation failed ({code}). No ticket ID is available. Please contact the Service Desk.",
                      tools=tool_calls, workflow="create_failed")
