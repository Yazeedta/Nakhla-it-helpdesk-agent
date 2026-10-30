"""Demo-only Gradio authentication and read-only MCP ticket history.

Keep accounts local and out of Git. This is not enterprise SSO or production IAM.
"""
from __future__ import annotations

import asyncio
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
USERS_FILE = ROOT / ".demo_users.json"
load_dotenv(ROOT / ".env")
ITERATIONS = 600_000


def _accounts() -> dict:
    if not USERS_FILE.is_file():
        return {}
    data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def verify_demo_user(username: str, password: str) -> bool:
    """Gradio auth callback. Usernames MUST be Nakhla employee email addresses."""
    email = (username or "").strip().lower()
    item = _accounts().get(email)
    if not isinstance(item, dict) or not item.get("enabled", True):
        return False
    try:
        salt = bytes.fromhex(item["salt"])
        saved = bytes.fromhex(item["password_hash"])
        rounds = int(item["iterations"])
        if rounds < 200_000:
            return False
        calculated = hashlib.pbkdf2_hmac("sha256", (password or "").encode(), salt, rounds)
        return hmac.compare_digest(saved, calculated)
    except (ValueError, KeyError, TypeError):
        return False


def require_user(request: gr.Request) -> str:
    email = (getattr(request, "username", None) or "").strip().lower()
    if not email or not verify_registered(email):
        raise gr.Error("Please sign in with your own Nakhla demo account.")
    return email


def verify_registered(email: str) -> bool:
    item = _accounts().get(email)
    return isinstance(item, dict) and item.get("enabled", True)


def signed_in_account(request: gr.Request):
    email = require_user(request)
    return gr.update(value=email, interactive=False), f"**Signed in as:** `{email}`"


async def _call_mcp(tool: str, args: dict) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "services.helpdesk_mcp"],
        env=dict(os.environ),
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(session.call_tool(tool, arguments=args), timeout=30)
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP {tool} failed")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        payload = structured
    else:
        parts = [p.text for p in result.content if getattr(p, "type", None) == "text"]
        if not parts:
            raise RuntimeError(f"MCP {tool} returned an empty response")
        payload = json.loads(parts[0])
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise RuntimeError(f"MCP {tool}: {payload.get('error', 'tool failure')}")
    if isinstance(payload, list):
        return {"tickets": payload}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response from MCP {tool}")
    return payload


def _tickets(payload: dict) -> list[dict]:
    for key in ("tickets", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [t for t in value if isinstance(t, dict)]
    d = payload.get("data")
    if isinstance(d, dict):
        return _tickets(d)
    return []


def _ticket_id(t: dict) -> str:
    return str(t.get("ticket_id") or t.get("id") or "")


def _clean(value: object, limit: int = 140) -> str:
    text = str(value if value is not None else "")[:limit]
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")


def _owner_matches(ticket: dict, email: str) -> bool:
    # NAKHLA_OWNER_CHECK_V2: Missing requester identity must NEVER grant access.
    actual = ticket.get("requester_email") or ticket.get("requester")
    if isinstance(actual, dict):
        actual = actual.get("email")
    return (
        isinstance(actual, str)
        and bool(actual.strip())
        and actual.strip().casefold() == email.strip().casefold()
    )


def _ticket_record(payload: dict) -> dict:
    # Handle common MCP response envelopes without trusting ticket summaries.
    ticket = payload.get("ticket")
    if isinstance(ticket, dict):
        return ticket
    data = payload.get("data")
    if isinstance(data, dict):
        nested = data.get("ticket")
        return nested if isinstance(nested, dict) else data
    return payload



async def my_ticket_history(request: gr.Request) -> str:
    email = require_user(request)
    try:
        payload = await _call_mcp("search_tickets", {"requester_email": email, "limit": 20})
        summaries = _tickets(payload)[:20]
    except Exception:
        return "**Ticket history unavailable.** Could not retrieve records through MCP."
    if not summaries:
        return f"No tickets found for `{_clean(email)}`."

    # Search summaries are NOT proof of ownership. Read each full ticket and
    # verify requester_email, failing closed when it is missing or different.
    verified = []
    seen = set()
    for summary in summaries:
        tid = _ticket_id(summary).strip().upper()
        if tid in seen or not re.fullmatch(r"[A-Z]{1,8}-[0-9]{1,12}", tid):
            continue
        seen.add(tid)
        try:
            record = _ticket_record(await _call_mcp("get_ticket", {"ticket_id": tid}))
        except Exception:
            continue
        if _ticket_id(record).strip().upper() != tid or not _owner_matches(record, email):
            continue
        verified.append(record)

    if not verified:
        return "**No tickets verified for your account.** The server did not confirm ticket ownership."
    lines = ["| Ticket | Subject | Status | Priority |", "|---|---|---|---|"]
    for t in verified:
        lines.append("| " + " | ".join([
            _clean(_ticket_id(t), 40),
            _clean(t.get("subject") or t.get("title") or "", 110),
            _clean(t.get("status") or "", 40),
            _clean(t.get("priority") or "", 20),
        ]) + " |")
    return "### My tickets (from MCP)\n" + "\n".join(lines)



async def my_ticket_details(ticket_id: str, request: gr.Request) -> str:
    email = require_user(request)
    tid = (ticket_id or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{1,8}-[0-9]{1,12}", tid):
        return "Enter a ticket ID such as `IT-1027`."
    try:
        # Confirm it appears in the signed-in user's filtered search.
        found = await _call_mcp("search_tickets", {"requester_email": email, "limit": 20})
        allowed = {_ticket_id(t).strip().upper() for t in _tickets(found)}
        if tid not in allowed:
            return "Ticket not found in your ticket history."
        # Crucial second check: verify ownership using the FULL record.
        ticket = _ticket_record(await _call_mcp("get_ticket", {"ticket_id": tid}))
        if _ticket_id(ticket).strip().upper() != tid or not _owner_matches(ticket, email):
            return "Ticket details cannot be verified for your account."
    except Exception:
        return "**Ticket details unavailable.** Could not verify this ticket via MCP."

    details = [f"### {_clean(tid, 40)}"]
    for key in ("subject", "status", "priority", "category", "department", "description", "attempted_steps"):
        if ticket.get(key) is not None:
            details.append(f"**{key.replace('_', ' ').title()}:** {_clean(ticket[key], 1500)}")
    notes = ticket.get("notes")
    if isinstance(notes, list):
        details.append("**Notes:**")
        details.extend("- " + _clean(n.get("note") if isinstance(n, dict) else n, 600) for n in notes[:20])
    return "\n\n".join(details)



def add_user(email: str) -> None:
    email = email.strip().lower()
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+", email):
        raise SystemExit("Use an actual employee email from services/data/employees.json")
    password = getpass.getpass("New demo password (minimum 12 characters): ")
    check = getpass.getpass("Confirm password: ")
    if len(password) < 12 or password != check:
        raise SystemExit("Passwords do not match or are shorter than 12 characters")
    users = _accounts()
    salt = secrets.token_bytes(16)
    users[email] = {
        "salt": salt.hex(),
        "password_hash": hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex(),
        "iterations": ITERATIONS,
        "enabled": True,
    }
    USERS_FILE.write_text(json.dumps(users, indent=2) + "\n", encoding="utf-8")
    print(f"Account created/updated for {email}. Password stored as a salted hash.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "add-user":
        add_user(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "list-users":
        for email in _accounts():
            print(email)
    else:
        print("Usage: python ui/demo_features.py add-user employee@nakhla.example")
        print("       python ui/demo_features.py list-users")
