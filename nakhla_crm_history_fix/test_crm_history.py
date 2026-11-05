"""Focused tests for descriptive, user-isolated ticket-memory search."""
import asyncio
from dataclasses import dataclass, field

from ticket_memory_router import _classify, handle_ticket_query

TARIQ = "tariq.hamdan@nakhla.example"
SARA = "sara.mikhail@nakhla.example"
TICKETS = {
    "IT-1002": {"ticket_id": "IT-1002", "requester_email": TARIQ,
                "subject": "NakhlaConnect NX-781 after laptop reimage",
                "description": "VPN client was old", "status": "resolved"},
    "IT-1021": {"ticket_id": "IT-1021", "requester_email": SARA,
                "subject": "NakhlaCRM sign-in fails with NID-512",
                "description": "This app is not assigned to you", "status": "resolved"},
}


@dataclass
class Session:
    last_error: str = "NX-781"
    last_category: str = "vpn"
    turns: list = field(default_factory=lambda: ["Earlier question about NX-781"])


async def mcp(name, args):
    if name == "search_tickets":
        # Deliberately leak both IDs at the search layer to verify full-record gate.
        return {"tickets": [{"ticket_id": x} for x in TICKETS]}
    if name == "get_ticket":
        return {"ticket": TICKETS[args["ticket_id"]]}
    raise AssertionError(name)


def ask(q, email):
    tools = []
    s = Session()
    answer = asyncio.run(handle_ticket_query(
        q, email, {"email": email, "account_status": "active"}, s, tools,
        mcp, lambda body, **kw: {"reply": body, **kw}))
    return answer, tools, s


def test_crm_signin_query_works_without_code():
    answer, tools, session = ask("Have I had a CRM sign-in problem before?", SARA)
    assert "IT-1021" in answer["reply"]
    assert "IT-1002" not in answer["reply"]
    assert {x["tool"] for x in tools} == {"search_tickets", "get_ticket"}
    assert session.last_error == "" and session.turns == []


def test_synonym_login_query_works():
    answer, _, _ = ask("Have I had a NakhlaCRM login problem before?", SARA)
    assert "IT-1021" in answer["reply"]


def test_no_cross_account_disclosure_on_crm_query():
    answer, _, _ = ask("Have I had a CRM sign-in problem before?", TARIQ)
    assert "IT-1021" not in answer["reply"]
    assert "No matching previous tickets" in answer["reply"]


def test_vague_before_has_no_fake_match():
    answer, tools, _ = ask("Have I had this problem before?", SARA)
    assert "Please give me the error code" in answer["reply"]
    assert "IT-1021" not in answer["reply"]


def test_existing_code_query_keeps_working():
    answer, _, _ = ask("Have I had NX-781 before?", TARIQ)
    assert "IT-1002" in answer["reply"]


def test_normal_ticket_creation_not_rerouted():
    assert _classify("Please open a support ticket for SWC-2001")[0] == ""
