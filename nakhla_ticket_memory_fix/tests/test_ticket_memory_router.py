import asyncio
from dataclasses import dataclass, field

from ticket_memory_router import _classify, handle_ticket_query

TARIQ = "tariq.hamdan@nakhla.example"
SARA = "sara.mikhail@nakhla.example"
RECORDS = {
    "IT-1002": {"ticket_id": "IT-1002", "requester_email": TARIQ, "subject": "NakhlaConnect NX-781 after reimage", "status": "resolved", "priority": "P2", "description": "VPN NX-781"},
    "IT-1021": {"ticket_id": "IT-1021", "requester_email": SARA, "subject": "CRM sign-in NID-512", "status": "resolved", "priority": "P4", "description": "NID-512"},
    "IT-2026": {"ticket_id": "IT-2026", "requester_email": SARA, "subject": "SWC-2001 license issue", "status": "open", "priority": "P3"},
}

@dataclass
class Session:
    last_error: str = "NX-781"
    last_category: str = "vpn"
    turns: list = field(default_factory=lambda: ["I have NX-781"])


def reply(text, **kwargs):
    return {"reply": text, **kwargs}


async def fake_mcp(tool, args):
    if tool == "search_tickets":
        # Deliberately return everyone to prove we VERIFY full owners.
        return {"tickets": [{"ticket_id": x} for x in RECORDS]}
    if tool == "get_ticket":
        return {"ok": True, "ticket": RECORDS[args["ticket_id"]]}
    raise AssertionError("Unexpected MCP tool")


def run(q, email=TARIQ, employee_override=None, mcp=fake_mcp):
    s, calls = Session(), []
    employee = employee_override if employee_override is not None else {"email": email, "account_status": "active"}
    result = asyncio.run(handle_ticket_query(q, email, employee, s, calls, mcp, reply))
    return result, calls, s


def test_classification():
    assert _classify("What are my open tickets?")[0] == "open"
    assert _classify("Can you show details of IT-1002?")[0] == "details"
    assert _classify("Show my previous IT tickets")[0] == "history"
    assert _classify("Have I had NX-781 before?")[0] == "similar"
    assert _classify("what are the key struct history")[0] == "clarify"
    assert _classify("Please open a support ticket for SWC-2001")[0] == ""
    assert _classify("Open a ticket for me")[0] == ""
    assert _classify("My VPN shows NX-503")[0] == ""


def test_details_ownership_and_tool_calls():
    yes, tools, s = run("Show me details of IT-1002", TARIQ)
    assert "NX-781" in yes["reply"] and any(t["tool"] == "get_ticket" for t in tools)
    no, tools, _ = run("Show me details of IT-1002", SARA)
    assert "not found" in no["reply"] and "NakhlaConnect" not in no["reply"]
    assert s.last_error == "" and s.last_category == "" and s.turns == []


def test_search_is_per_user_even_if_search_mcp_leaks_other_tickets():
    tariq, tools, _ = run("Show my ticket history", TARIQ)
    assert "IT-1002" in tariq["reply"] and "IT-1021" not in tariq["reply"]
    assert any(x["tool"] == "search_tickets" for x in tools)
    assert any(x["tool"] == "get_ticket" for x in tools)
    sara, _, _ = run("Show my ticket history", SARA)
    assert "IT-1002" not in sara["reply"] and "IT-1021" in sara["reply"]


def test_open_ticket_and_long_term_similar():
    sara, _, _ = run("What are my open tickets?", SARA)
    assert "IT-2026" in sara["reply"] and "IT-1021" not in sara["reply"]
    tariq, _, _ = run("What are my open tickets?", TARIQ)
    assert "No open tickets" in tariq["reply"]
    prior, _, _ = run("Have I had the NX-781 error before?", TARIQ)
    assert "IT-1002" in prior["reply"]


def test_read_failure_fails_closed():
    async def broken(name, args):
        raise RuntimeError("MCP offline")
    r, tools, _ = run("Show my ticket history", mcp=broken)
    assert "unavailable" in r["reply"] and tools[0]["ok"] is False


def test_profile_unverified_fails_closed():
    result, tools, _ = run("Show details of IT-1002", employee_override={"email": SARA})
    assert "unavailable" in result["reply"] and not tools


def test_normal_help_untouched():
    result, tools, s = run("My VPN shows NX-781")
    assert result is None and not tools and s.last_error == "NX-781"


def test_no_ticket_note_injection():
    record = dict(RECORDS["IT-1002"])
    record["description"] = "<script>alert('x')</script>"
    async def mcp(tool, args):
        return {"ticket": record}
    result, _, _ = run("Show details of IT-1002", mcp=mcp)
    assert "<script>" not in result["reply"] and "&lt;script&gt;" in result["reply"]
