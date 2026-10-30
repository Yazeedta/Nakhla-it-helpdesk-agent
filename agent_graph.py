"""Nakhla IT Helpdesk LangGraph workflow.

Reads KB data via retrieval.py and service incidents via read-only MCP.
Proposes tickets but NEVER creates or modifies them.
Supports demonstration-only human review via LangGraph interrupt.
Run with build_graph().ainvoke({...}) because check_service is asynchronous.
For pausing/resuming a review, supply a checkpointer and a thread_id.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from policy_decision import decide_policy
from retrieval import search

load_dotenv()


class AgentState(TypedDict):
    # Inputs
    question: str
    department: NotRequired[str]
    affected_users: NotRequired[int]
    nakhlakey_on_lost_phone: NotRequired[bool]
    gateway: NotRequired[str]
    employee_email: NotRequired[str]  # Demo input; NOT proof of identity

    # Intermediate results
    triage: NotRequired[dict]
    employee: NotRequired[dict | None]
    employee_lookup: NotRequired[dict]
    retrieved_documents: NotRequired[list[dict]]
    policy_decision: NotRequired[dict]
    service_status: NotRequired[dict]
    incident_match: NotRequired[dict | None]
    incident_applicability: NotRequired[dict]
    final_answer: NotRequired[str]
    llm_draft: NotRequired[str]

    # Verification and citations
    verified_answer: NotRequired[str]
    verification: NotRequired[dict]
    source_refs: NotRequired[list[str]]
    cited_answer: NotRequired[str]
    source_validation: NotRequired[dict]

    # Approval proposal only; no write operations in this graph
    ticket_proposal: NotRequired[dict | None]
    approval_status: NotRequired[str]
    human_review_decision: NotRequired[str]


def triage(state: AgentState) -> dict:
    # Identify a VPN question and its NX error code.
    question = state["question"]
    match = re.search(r"\bNX-\d{3}\b", question, re.IGNORECASE)
    error_code = match.group(0).upper() if match else None
    category = "vpn" if ("vpn" in question.lower() or error_code) else "unknown"
    return {"triage": {"category": category, "error_code": error_code}}



async def lookup_employee_record(state: AgentState) -> dict:
    """Read directory attributes by email; this is NOT user authentication."""
    email = (state.get("employee_email") or "").strip().lower()
    if not email:
        return {
            "employee": None,
            "employee_lookup": {
                "status": "email_required",
                "identity_authenticated": False,
            },
        }

    # This limits malformed or injected addresses before calling the API.
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+", email):
        return {
            "employee": None,
            "employee_lookup": {
                "status": "invalid_email",
                "identity_authenticated": False,
            },
        }

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "services.helpdesk_mcp"],
        env=os.environ.copy(),
    )
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "lookup_employee", arguments={"email": email}
                )
                if result.isError:
                    raise RuntimeError("MCP employee lookup failed")
                texts = [part.text for part in result.content if part.type == "text"]
                if not texts:
                    raise ValueError("Empty MCP employee lookup response")
                payload = json.loads(texts[0])
    except Exception:
        # Fail closed: a directory outage must not fall back to user-provided
        # department/priority information.
        return {
            "employee": None,
            "employee_lookup": {
                "status": "lookup_unavailable",
                "identity_authenticated": False,
            },
        }

    employee = payload.get("employee") if isinstance(payload, dict) and payload.get("ok") else None
    if not isinstance(employee, dict) or str(employee.get("email") or "").lower() != email:
        return {
            "employee": None,
            "employee_lookup": {
                "status": "not_found",
                "identity_authenticated": False,
            },
        }

    account_status = (employee.get("account_status") or "").lower()
    if account_status in {"disabled", "inactive", "locked", "suspended", "terminated"}:
        return {
            "employee": None,
            "employee_lookup": {
                "status": "inactive_account",
                "identity_authenticated": False,
            },
        }

    return {
        "employee": employee,
        "employee_lookup": {
            "status": "found",
            # Looking someone up by a supplied address does not authenticate them.
            "identity_authenticated": False,
        },
    }

def retrieve(state: AgentState) -> dict:
    # Use the existing RAG retrieval implementation.
    return {"retrieved_documents": search(state["question"], top_k=3)}


def check_policy(state: AgentState) -> dict:
    # Apply the policy rules defined in policy_decision.py.
    lookup = state.get("employee_lookup") or {}
    employee = state.get("employee") or {}
    # Ignore any department asserted directly in the user/graph input.
    department = employee.get("department") if lookup.get("status") == "found" else None
    decision = decide_policy(
        query=state["question"],
        department=department,
        affected_users=state.get("affected_users"),
        nakhlakey_on_lost_phone=state.get("nakhlakey_on_lost_phone"),
    )
    return {"policy_decision": decision}


async def check_service(state: AgentState) -> dict:
    # Fetch live helpdesk service status via the read-only MCP tool.
    triage_data = state.get("triage") or {}
    policy = state.get("policy_decision") or {}
    should_check = (
        triage_data.get("category") == "vpn"
        or bool(policy.get("check_service_status"))
    )
    if not should_check:
        return {"service_status": {"skipped": True}}

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "services.helpdesk_mcp"],
        env=os.environ.copy(),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_service_status", arguments={})
            if result.isError:
                raise RuntimeError("MCP get_service_status returned an error")
            texts = [part.text for part in result.content if part.type == "text"]
            if not texts:
                raise ValueError("MCP get_service_status returned no text")
            payload = json.loads(texts[0])
            if not payload.get("ok") or not isinstance(payload.get("status"), dict):
                raise RuntimeError("Invalid helpdesk service status response")
    return {"service_status": payload["status"]}


def analyze_incident(state: AgentState) -> dict:
    # Match an active incident by service and exact error code.
    status = state.get("service_status") or {}
    triage_data = state.get("triage") or {}
    category = triage_data.get("category")
    error_code = triage_data.get("error_code")
    active_ids = set(status.get("active_incidents", []))

    for incident in status.get("incidents", []):
        if (
            incident.get("id") in active_ids
            and incident.get("status") == "active"
            and incident.get("service") == category
            and error_code is not None
            and error_code in incident.get("error_codes", [])
        ):
            return {
                "incident_match": {
                    "id": incident.get("id"),
                    "title": incident.get("title"),
                    "affected": incident.get("affected"),
                    "workaround": incident.get("workaround"),
                    "instructions": incident.get("instructions"),
                }
            }
    return {"incident_match": None}


def verify_impact(state: AgentState) -> dict:
    # A matching error is not proof that this employee is affected.
    incident = state.get("incident_match")
    if not incident:
        return {"incident_applicability": {"status": "no_incident_match"}}

    employee = state.get("employee") or {}
    directory_gateway = (
        employee.get("vpn_gateway")
        if (state.get("employee_lookup") or {}).get("status") == "found"
        else None
    )
    # A manually supplied current gateway takes precedence over the directory
    # default, because employees can switch gateways. This is still unverified.
    gateway = (state.get("gateway") or directory_gateway or "").strip().upper()
    if not gateway:
        return {
            "incident_applicability": {
                "status": "needs_gateway",
                "question": "Which VPN gateway are you using?",
            }
        }

    affected_gateways = {
        code.upper()
        for code in re.findall(
            r"\bGW-\d+\b", incident.get("affected") or "", re.IGNORECASE
        )
    }
    if not affected_gateways:
        status = "needs_verification"
    elif gateway in affected_gateways:
        status = "affected"
    else:
        status = "gateway_not_listed"

    return {
        "incident_applicability": {
            "status": status,
            "gateway": gateway,
            "incident_id": incident["id"],
        }
    }


def resolve(state: AgentState) -> dict:
    # Build a conservative rule-based answer before using an LLM.
    policy = state.get("policy_decision") or {}
    impact = state.get("incident_applicability") or {}
    incident = state.get("incident_match") or {}

    if policy.get("needs_clarification"):
        warnings = policy.get("blocked_actions") or []
        return {"final_answer": "\n".join(warnings + [policy["needs_clarification"]])}

    if policy.get("allow_vpn_troubleshooting") is False:
        return {
            "final_answer": (
                "This VPN request requires special security handling. "
                "Do not use general VPN troubleshooting steps. "
                "Refer the case to Security Operations. "
                "Any ticket creation requires user approval."
            )
        }

    if impact.get("status") == "affected":
        return {
            "final_answer": (
                f"Active incident: {incident['id']}\n"
                f"Problem: {incident['title']}\n\n"
                f"Workaround: {incident.get('workaround') or 'Not available'}\n\n"
                f"Instructions: {incident.get('instructions') or 'No instructions'}"
            )
        }

    if impact.get("status") == "needs_gateway":
        return {"final_answer": impact["question"]}

    if impact.get("status") == "gateway_not_listed":
        return {
            "final_answer": (
                "Your gateway is not listed as affected by this incident. "
                "We need to investigate your issue separately."
            )
        }

    return {
        "final_answer": (
            "I found information about your problem, but more checks are "
            "needed before recommending an action."
        )
    }


def generate_answer(state: AgentState) -> dict:
    # Rewrite only the established incident workaround; keep other cases exact.
    original = state.get("final_answer") or ""
    policy = state.get("policy_decision") or {}
    impact = state.get("incident_applicability") or {}

    protected = (
        bool(policy.get("needs_clarification"))
        or policy.get("allow_vpn_troubleshooting") is False
        or impact.get("status") != "affected"
    )
    if protected:
        return {"llm_draft": original}

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an IT helpdesk assistant. Rewrite the supplied answer "
                    "in clear, helpful English. Use only facts in the supplied "
                    "answer. Preserve incident IDs, instructions, gateway names, "
                    "restrictions and warnings. Do not add troubleshooting steps. "
                    "Do not suggest creating tickets when forbidden. "
                    "Do not invent information."
                ),
            },
            {"role": "user", "content": original},
        ],
    )
    draft = response.choices[0].message.content
    return {"llm_draft": draft or original}


def verify_answer(state: AgentState) -> dict:
    # Check preservation of key identifiers and ticket restrictions.
    original = (state.get("final_answer") or "").strip()
    draft = (state.get("llm_draft") or "").strip()
    policy = state.get("policy_decision") or {}
    impact = state.get("incident_applicability") or {}
    incident = state.get("incident_match") or {}
    problems = []

    if not draft:
        problems.append("Empty LLM answer")

    protected = (
        bool(policy.get("needs_clarification"))
        or policy.get("allow_vpn_troubleshooting") is False
        or impact.get("status") != "affected"
    )
    if protected:
        if draft != original:
            problems.append("Protected answer was changed")
    else:
        incident_id = incident.get("id")
        if incident_id and incident_id.lower() not in draft.lower():
            problems.append("Missing incident ID")

        original_gateways = set(re.findall(r"\bGW-\d+\b", original.upper()))
        draft_gateways = set(re.findall(r"\bGW-\d+\b", draft.upper()))
        if original_gateways != draft_gateways:
            problems.append("Gateway references changed")

        instructions = (incident.get("instructions") or "").lower()
        if "do not open individual tickets" in instructions:
            no_ticket = re.search(
                r"\b(?:do not|don't|never|must not)\s+"
                r"(?:\w+\s+){0,3}"
                r"(?:open|create|submit|file)\s+"
                r"(?:\w+\s+){0,4}tickets?\b",
                draft,
                re.IGNORECASE,
            )
            if not no_ticket:
                problems.append("Ticket restriction missing")

            # Look for later ticket-creation instructions contradicting the ban.
            clauses = re.split(
                r"(?<=[.!?])\s+|\b(?:however|but|instead)\b[:,]?\s*",
                draft,
                flags=re.IGNORECASE,
            )
            ticket_action = re.compile(
                r"\b(?:open|create|submit|file|raise)\s+"
                r"(?:(?:a|an|new|another|individual|separate|support)\s+){0,4}"
                r"tickets?\b",
                re.IGNORECASE,
            )
            negation = re.compile(
                r"\b(?:do not|don't|never|must not|should not|avoid)\b",
                re.IGNORECASE,
            )
            for clause in clauses:
                if ticket_action.search(clause) and not negation.search(clause):
                    problems.append("Conflicting ticket instruction")
                    break

        if "subscribe" in instructions:
            if not ("subscribe" in draft.lower() and "status page" in draft.lower()):
                problems.append("Subscription instructions missing")

    passed = not problems
    return {
        "verified_answer": draft if passed else original,
        "verification": {
            "passed": passed,
            "problems": problems,
            "used_fallback": not passed,
        },
    }


def attach_sources(state: AgentState) -> dict:
    # Cite policy references or the incident used to decide the answer.
    answer = state.get("verified_answer") or state.get("final_answer") or ""
    policy = state.get("policy_decision") or {}
    impact = state.get("incident_applicability") or {}
    incident = state.get("incident_match") or {}
    sources = []

    if (policy.get("needs_clarification")
            or policy.get("allow_vpn_troubleshooting") is False):
        sources.extend(policy.get("policy_refs") or [])
    elif impact.get("status") in ("affected", "gateway_not_listed"):
        if incident.get("id"):
            sources.append(incident["id"])

    sources = list(dict.fromkeys(sources))
    cited_answer = answer
    if sources:
        cited_answer += "\n\nSources: " + ", ".join(sources)
    return {"source_refs": sources, "cited_answer": cited_answer}


def validate_citations(state: AgentState) -> dict:
    # Check source IDs exist; this is not claim-by-claim fact checking.
    refs = state.get("source_refs") or []
    problems = []
    kb_ids = set()

    impact = state.get("incident_applicability") or {}
    answer = (state.get("cited_answer") or "").strip()
    is_clarification = (
        impact.get("status") == "needs_gateway"
        and answer == (impact.get("question") or "").strip()
        and bool(answer)
    )
    if not refs and not is_clarification:
        problems.append("Missing supporting sources")

    chunks_path = Path(__file__).resolve().parent / "rag_store" / "chunks.json"
    try:
        with chunks_path.open(encoding="utf-8") as file:
            chunks = json.load(file)
        kb_ids = {
            item["document_id"]
            for item in chunks
            if isinstance(item, dict) and "document_id" in item
        }
    except (OSError, ValueError, TypeError, KeyError):
        problems.append("Could not verify knowledge base sources")

    service_status = state.get("service_status") or {}
    active_ids = set(service_status.get("active_incidents", []))
    valid_incidents = {
        item.get("id")
        for item in service_status.get("incidents", [])
        if item.get("status") == "active" and item.get("id") in active_ids
    }

    for ref in refs:
        if ref.startswith(("KB-", "SP-")):
            valid = ref in kb_ids
        elif ref.startswith("INC-"):
            valid = ref in valid_incidents
        else:
            valid = False
        if not valid:
            problems.append(f"Unverified source: {ref}")

    passed = not problems
    return {
        "source_validation": {
            "passed": passed,
            "checked_sources": refs,
            "problems": problems,
        },
        "cited_answer": (
            state.get("cited_answer", "")
            if passed
            else "I could not verify the supporting sources. Please contact IT support."
        ),
    }


def prepare_approval(state: AgentState) -> dict:
    # Prepare a ticket proposal ONLY. Never call create_ticket here.
    policy = state.get("policy_decision") or {}
    proposal = policy.get("proposed_ticket")
    if not proposal:
        return {"ticket_proposal": None, "approval_status": "not_required"}

    if policy.get("needs_clarification"):
        return {"ticket_proposal": None, "approval_status": "needs_clarification"}

    impact = state.get("incident_applicability") or {}
    incident = state.get("incident_match") or {}
    incident_blocks_tickets = (
        impact.get("status") == "affected"
        and "do not open individual tickets"
        in (incident.get("instructions") or "").lower()
    )
    if incident_blocks_tickets:
        return {"ticket_proposal": None, "approval_status": "blocked_active_incident"}

    # Do not propose while a matching incident still requires verification.
    if incident and impact.get("status") in ("needs_gateway", "needs_verification"):
        return {"ticket_proposal": None, "approval_status": "needs_clarification"}

    verification = state.get("verification") or {}
    source_validation = state.get("source_validation") or {}
    if not verification.get("passed") or not source_validation.get("passed"):
        return {"ticket_proposal": None, "approval_status": "blocked_verification"}

    # Directory lookup is required before requesting approval for a ticket.
    # The department supplied by the caller must never authorize a ticket.
    lookup = state.get("employee_lookup") or {}
    employee = state.get("employee")
    if lookup.get("status") != "found" or not isinstance(employee, dict):
        return {"ticket_proposal": None, "approval_status": "blocked_employee_lookup"}

    # A successful lookup only identifies a directory record. It does NOT
    # authenticate the person chatting with the agent. Our current lookup node
    # explicitly sets identity_authenticated=False, so all requests remain
    # blocked until we integrate trusted server-side authentication.
    # This flag alone must NOT be used to authorize future write tools.
    if lookup.get("identity_authenticated") is not True:
        return {"ticket_proposal": None, "approval_status": "blocked_unauthenticated"}

    ticket = {
        "category": proposal.get("category"),
        "priority": proposal.get("priority"),
        "team": proposal.get("team"),
        "summary": state["question"],
    }
    return {"ticket_proposal": ticket, "approval_status": "pending_approval"}


def human_review(state: AgentState) -> dict:
    """Demo-only human review; it NEVER executes any helpdesk WRITE tool.

    A pending draft must have passed prepare_approval first in the real graph.
    A resume value is NOT proof of identity or server-side authorization.
    """
    if state.get("approval_status") != "pending_approval":
        return {"human_review_decision": "not_requested"}

    draft = state.get("ticket_proposal")
    if not isinstance(draft, dict) or not draft:
        return {"human_review_decision": "blocked_missing_draft"}

    response = interrupt({
        "action": "review_ticket_demo_only",
        "ticket_draft": draft,
        "message": "Review this exact ticket proposal. Approve or reject?",
        "warning": "This approval is a demonstration. No ticket will be created.",
    })

    if response == "approve":
        return {"human_review_decision": "approved_demo_only"}
    return {"human_review_decision": "rejected"}


def build_graph(*, checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage)
    graph.add_node("employee_lookup", lookup_employee_record)
    graph.add_node("retriever", retrieve)
    graph.add_node("policy", check_policy)
    graph.add_node("check_service", check_service)
    graph.add_node("analyze_incident", analyze_incident)
    graph.add_node("verify_impact", verify_impact)
    graph.add_node("resolver", resolve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verifier", verify_answer)
    graph.add_node("attach_sources", attach_sources)
    graph.add_node("validate_citations", validate_citations)
    graph.add_node("approval_gate", prepare_approval)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "employee_lookup")
    graph.add_edge("employee_lookup", "retriever")
    graph.add_edge("retriever", "policy")
    graph.add_edge("policy", "check_service")
    graph.add_edge("check_service", "analyze_incident")
    graph.add_edge("analyze_incident", "verify_impact")
    graph.add_edge("verify_impact", "resolver")
    graph.add_edge("resolver", "generate_answer")
    graph.add_edge("generate_answer", "verifier")
    graph.add_edge("verifier", "attach_sources")
    graph.add_edge("attach_sources", "validate_citations")
    graph.add_edge("validate_citations", "approval_gate")
    graph.add_edge("approval_gate", "human_review")
    graph.add_edge("human_review", END)

    # Keep existing read-only smoke tests compatible without a thread_id.
    # A persistent thread_id + checkpointer are required when interrupting.
    return graph.compile(checkpointer=checkpointer)
