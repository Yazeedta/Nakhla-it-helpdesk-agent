"""Read-only, authenticated-UI-facing MCP ticket router for the Nakhla demo.

This augments the existing course backend; it does not edit the simulated kit.
IMPORTANT: the course FastAPI backend trusts a client-supplied email and is
NOT production authenticated. Keep it bound to 127.0.0.1 and do not publish it.
"""
from __future__ import annotations

import html
import re
from typing import Any

_TICKET = re.compile(r"\bIT-\d{1,12}\b", re.IGNORECASE)
_ERROR = re.compile(r"\b(?:NX|SWC|NID|CRM)-\d{3,5}\b", re.IGNORECASE)
_OPEN = frozenset({"open", "new", "pending", "in_progress", "in progress", "in-progress", "assigned", "waiting", "reopened", "on_hold", "on hold"})
_CODE = re.compile(r"\b(?:NX|SWC|NID|CRM)-\d{3,5}\b", re.IGNORECASE)


def _classify(message: str) -> tuple[str, str | None]:
    """Only route ticket *reads*. Creation requests remain with the old approval flow."""
    q = " ".join((message or "").lower().split())
    tid = _TICKET.search(q)
    # Ticket details for a specific ID; use real MCP read, never fabricate text.
    if tid and (
        any(x in q for x in ("ticket", "case", "details", "detail", "status", "show", "about", "notes", "what happened", "history", "previous"))
    ):
        return ("details", tid.group(0).upper())

    # Do not steal requests to CREATE a ticket. E.g. "Please open a support ticket".
    if re.search(r"\b(?:open|create|raise|submit|file)\s+(?:a|an|new|support|another)\s+(?:support\s+)?ticket\b", q):
        return ("", None)

    # NAKHLA_FINANCE_NEW_TICKET_ROUTING_V1
    # A newly reported error that mentions a possible ticket belongs to the
    # normal troubleshooting/approval workflow, NOT ticket-history retrieval.
    # Preserve explicit queries about old/current tickets and ticket IDs.
    if _ERROR.search(q) and re.search(r"\btickets?\b", q):
        asking_to_read_tickets = bool(
            _TICKET.search(q)
            or re.search(r"\b(?:show|list|find|search|view|check)\b.{0,60}\b(?:tickets?|cases?)\b", q)
            or re.search(r"\b(?:my|our|your|previous|past|existing|old|earlier|prior)\s+(?:[a-z]+\s+){0,3}tickets?\b", q)
            or re.search(r"\b(?:tickets?|cases?)\s+(?:history|status|details)\b", q)
            or re.search(r"\b(?:before|previously|last time|earlier|already)\b", q)
        )
        if not asking_to_read_tickets:
            return ("", None)

    talks_tickets = bool(re.search(r"\b(?:tickets?|cases?|support requests?|incidents?)\b", q))
    historical = bool(re.search(r"\b(?:previous|prior|past|history|historical|before|last time|similar|earlier|again)\b", q))
    ownership = bool(re.search(r"\b(?:my|mine|i |me |for me|mine\b)\b", q))
    status_words = bool(re.search(r"\b(?:open|active|unresolved|pending|still waiting)\b", q))
    if talks_tickets and (status_words or historical or ownership or any(x in q for x in ("list", "show", "find", "all tickets"))):
        if historical and (_CODE.search(q) or re.search(r"\b(?:same|similar|this problem|this issue|before)\b", q)):
            return ("similar", None)
        return ("open" if status_words and not historical else "history", None)
    if historical and (re.search(r"\b(?:have i had|had this|same issue|this issue|this problem|trouble|error|problem|vpn)\b", q) or _CODE.search(q)):
        return ("similar", None)
    # "key struct history" (unclear speech input) must ask, not hallucinate a KB fix.
    if "history" in q and not talks_tickets:
        return ("clarify", None)
    return ("", None)


def _nested_ticket(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if payload.get("ok") is False or payload.get("error"):
        raise RuntimeError("MCP ticket operation failed")
    ticket = payload.get("ticket")
    if isinstance(ticket, dict):
        return ticket
    data = payload.get("data")
    if isinstance(data, dict):
        return _nested_ticket(data)
    return payload


def _tickets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and (payload.get("ok") is False or payload.get("error")):
        raise RuntimeError("MCP ticket search failed")
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for k in ("tickets", "results", "items"):
        v = payload.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    if isinstance(payload.get("data"), (dict, list)):
        return _tickets(payload["data"])
    return []


def _id(t: dict) -> str:
    return str(t.get("ticket_id") or t.get("id") or "").strip().upper()


def _owner(t: dict, email: str) -> bool:
    actual = t.get("requester_email") or t.get("requester")
    if isinstance(actual, dict):
        actual = actual.get("email")
    return isinstance(actual, str) and bool(actual.strip()) and actual.strip().casefold() == email.casefold()


def _s(value: Any, n: int = 140) -> str:
    v = str(value if value is not None else "")[:n]
    return html.escape(v, quote=True).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "'")


def _status(record: dict) -> str:
    return str(record.get("status") or "unknown").strip().lower()


# NAKHLA_DESCRIPTIVE_TICKET_SEARCH_V2
# Deterministic token overlap, not an LLM or a general semantic-search index.
# Normalizes common sign-in terms and Nakhla product names.
_DESCRIPTIVE_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "i", "my", "me", "mine",
    "have", "had", "has", "did", "do", "does", "am", "been", "before",
    "previous", "prior", "past", "history", "historical", "last", "time",
    "problem", "problems", "issue", "issues", "error", "errors", "ticket",
    "tickets", "support", "case", "cases", "similar", "same", "this", "that",
    "what", "which", "when", "how", "with", "for", "of", "any", "about",
    "get", "got", "having", "trouble", "again", "earlier", "ever", "could",
    "you", "show", "find", "me", "had", "it", "please", "something", "related",
    "on", "at", "to", "and", "in", "or", "from", "not", "still", "there",
})


def _descriptive_terms(value: Any) -> set[str]:
    text = str(value or "").casefold()
    text = re.sub(r"\bsign[\s_-]*in\b", " signin ", text)
    text = re.sub(r"\blog[\s_-]*in\b", " signin ", text)
    text = re.sub(r"\b(?:login|logon|authentication|authenticated)\b", " signin ", text)
    tokens = re.findall(r"[a-z][a-z0-9]*", text[:5000])
    terms = set()
    for t in tokens:
        if t in _DESCRIPTIVE_STOP:
            continue
        if t == "nakhlacrm":
            t = "crm"
        elif t == "nakhlaconnect":
            t = "vpn"
        elif t in {"signing", "signin", "logged"}:
            t = "signin"
        if len(t) > 2:
            terms.add(t)
    return terms


def _descriptive_match(terms: set[str], record: dict[str, Any]) -> bool:
    description = " ".join(str(record.get(k) or "") for k in (
        "subject", "title", "description", "category", "attempted_steps"))
    matching = terms.intersection(_descriptive_terms(description))
    # Two independent terms when available; a single precise product term is
    # enough if that is all the user supplied (e.g. "previous CRM problems").
    return len(matching) >= min(2, len(terms))


async def handle_ticket_query(message, email, employee, session, tools, mcp_call, reply_fn):
    """Return a normal API chat response for ticket reads, or None otherwise.

    Do not treat search summaries as ownership proof; get full ticket record and
    match requester_email. No ticket data is used as a model instruction.
    """
    mode, ticket_id = _classify(message)
    if not mode:
        return None

    # Clear stale error/issue context once an unrelated history intent is recognized.
    session.last_error = ""
    session.last_category = ""
    session.turns.clear()

    def answer(body: str):
        return reply_fn(body, tools=tools, workflow="ticket_memory")

    if mode == "clarify":
        return answer("Do you mean your previous IT support tickets, or the change history of a knowledge-base document?")

    # Directory lookup does NOT authenticate an employee. This demo path is
    # acceptable only behind the authenticated Gradio UI on loopback.
    if not isinstance(employee, dict) or not email or str(employee.get("email") or "").strip().casefold() != str(email).strip().casefold():
        return answer("I could not verify an employee record for this request. Ticket history is unavailable.")
    if str(employee.get("account_status") or "active").lower() not in {"active", "enabled"}:
        return answer("Ticket history is unavailable for this account.")

    async def tool_call(name: str, args: dict) -> Any:
        try:
            result = await mcp_call(name, args)
            # Treat error envelopes as failures even if tool transport succeeded.
            if isinstance(result, dict) and (result.get("ok") is False or result.get("error")):
                raise RuntimeError("tool error")
            tools.append({"tool": name, "ok": True})
            return result
        except Exception:
            tools.append({"tool": name, "ok": False})
            raise

    async def owned_ticket(tid: str) -> dict:
        payload = await tool_call("get_ticket", {"ticket_id": tid})
        record = _nested_ticket(payload)
        if _id(record) == tid and _owner(record, email):
            return record
        return {}

    if mode == "details":
        try:
            record = await owned_ticket(ticket_id)
        except Exception:
            return answer("Ticket details are temporarily unavailable. I couldn't verify them through MCP.")
        if not record:
            return answer("That ticket was not found in your own ticket history, or its ownership could not be verified.")
        fields = [("Ticket", ticket_id), ("Subject", record.get("subject") or record.get("title")),
                  ("Status", record.get("status")), ("Priority", record.get("priority")),
                  ("Category", record.get("category")), ("Department", record.get("department")),
                  ("Description", record.get("description")), ("Attempted steps", record.get("attempted_steps"))]
        content = [f"**{name}:** {_s(value, 700 if name in ('Description', 'Attempted steps') else 160)}" for name, value in fields if value is not None]
        notes = record.get("notes")
        if isinstance(notes, list) and notes:
            content.append("**Recent notes:** " + "; ".join(_s(n.get("note") if isinstance(n, dict) else n, 250) for n in notes[-3:]))
        return answer("\n\n".join(content))

    try:
        # Small bounded list with per-record ownership verification. When a real
        # backend adds paging, extend this to paginate rather than claiming all.
        results = await tool_call("search_tickets", {"requester_email": email, "limit": 20})
        summaries = _tickets(results)[:20]
    except Exception:
        return answer("Ticket history is temporarily unavailable. The MCP search failed.")

    verified: list[dict] = []
    seen: set[str] = set()
    for short in summaries:
        tid = _id(short)
        if not re.fullmatch(r"[A-Z]{1,8}-[0-9]{1,12}", tid) or tid in seen:
            continue
        seen.add(tid)
        try:
            record = await owned_ticket(tid)
        except Exception:
            continue
        if record:
            verified.append(record)

    if mode == "open":
        verified = [record for record in verified if _status(record) in _OPEN]
    elif mode == "similar":
        codes = {x.upper() for x in _ERROR.findall(message)}
        if codes:
            verified = [r for r in verified if any(c in str(r.get(k) or "").upper() for c in codes for k in ("subject", "description", "category"))]
        else:
            # Users may describe a previous issue without knowing its error code.
            # Match only VERIFIED owned records; never broaden access rights.
            query_terms = _descriptive_terms(message)
            if not query_terms:
                return answer("Please give me the error code or problem description to search similar past tickets (for example NX-781).")
            verified = [r for r in verified if _descriptive_match(query_terms, r)]

    if not verified:
        if mode == "open":
            return answer("No open tickets were verified among the first 20 returned for your account.")
        if mode == "similar":
            return answer("No matching previous tickets were verified for your account in the first 20 results.")
        return answer("No previous tickets could be verified for your account from the returned MCP records.")

    lines = ["| Ticket ID | Subject | Status | Priority |", "|---|---|---|---|"]
    for t in verified:
        lines.append(f"| {_s(_id(t), 24)} | {_s(t.get('subject') or t.get('title') or '', 130)} | {_s(t.get('status'), 40)} | {_s(t.get('priority'), 16)} |")
    intro = {"history": "Here are your verified previous support tickets:",
             "open": "These are your verified open support tickets:",
             "similar": "I found these similar cases in your ticket history (past cases do not override current KB policy):"}[mode]
    return answer(intro + "\n\n" + "\n".join(lines) + "\n\nYou can ask for details of a specific ticket ID.")
