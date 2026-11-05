"""Regression tests for S2-10 Finance ticket creation and ticket-memory reads."""
from ticket_memory_router import _classify


def test_finance_new_ticket_does_not_route_to_history():
    examples = [
        "I'm in Finance. NakhlaConnect shows NX-403 when I connect from home. If it needs a ticket, please open one.",
        "I'm in Finance. NakhlaConnect shows NX-403 when I connect from home. If it needs a ticket, create one for me.",
        "I'm in Finance. NakhlaConnect shows NX-403 when I connect from home. If it needs a ticket, log one for me.",
        "I'm in Finance. NakhlaConnect shows NX-403 when I connect from home. If it needs a ticket, request my approval.",
        "Finance NX-403 VPN blocked. Please create a security ticket.",
    ]
    for message in examples:
        assert _classify(message)[0] == "", (message, _classify(message))


def test_ticket_history_still_works():
    cases = [
        ("What are my open tickets?", "open"),
        ("Show my ticket history.", "history"),
        ("Show me the details of ticket IT-1002.", "details"),
        ("Have I had NX-781 before?", "similar"),
        ("Have I had a CRM sign-in problem before?", "similar"),
        ("What are my open tickets related to NX-781?", "open"),
    ]
    for message, expected in cases:
        assert _classify(message)[0] == expected, (message, _classify(message))
