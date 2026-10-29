
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from agent_graph import graph


load_dotenv()

# Load the official ticket schema
SCHEMA_PATH = Path("dataset/schemas/ticket.schema.json")

schema = json.loads(
    SCHEMA_PATH.read_text(encoding="utf-8")
)

validator = Draft202012Validator(schema)

app = FastAPI(title="Nakhla Ticket Approval Prototype")

# Store pending ticket drafts
pending_actions = {}

# Prevent concurrent approvals from creating duplicate tickets
approval_lock = asyncio.Lock()


# Request models
class DraftRequest(BaseModel):
    email: str
    question: str
    attempted_steps: list[str] = Field(
        min_length=1,
        max_length=20
    )
    affected_users: int = Field(
        default=1,
        ge=1,
        le=1000
    )


class ActionRequest(BaseModel):
    action_id: str


# Validate the ticket against the official JSON schema
def validate_ticket(ticket: dict):
    errors = list(validator.iter_errors(ticket))

    if errors:
        message = "; ".join(
            error.message for error in errors[:3]
        )

        raise HTTPException(
            status_code=422,
            detail=f"Invalid ticket draft: {message}"
        )


# Create a real ticket through MCP ONLY after approval
async def create_ticket_after_approval(
    ticket: dict,
    action_id: str
):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "services.helpdesk_mcp"],
        env=os.environ.copy()
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover the actual create_ticket tool schema
            available = await session.list_tools()

            tool = next(
                (
                    item for item in available.tools
                    if item.name == "create_ticket"
                ),
                None
            )

            if tool is None:
                raise RuntimeError(
                    "create_ticket MCP tool is unavailable"
                )

            properties = tool.inputSchema.get(
                "properties", {}
            )

            if "idempotency_key" not in properties:
                raise RuntimeError(
                    "MCP create_ticket lacks idempotency_key"
                )

            # Send only arguments supported by MCP
            arguments = {
                key: value
                for key, value in ticket.items()
                if key in properties
            }

            # Use the action ID as the idempotency key
            arguments["idempotency_key"] = action_id

            result = await session.call_tool(
                "create_ticket",
                arguments=arguments
            )

            if result.isError:
                raise RuntimeError(
                    "MCP create_ticket returned an error"
                )

            # Accept a ticket ID only from the MCP response
            for item in result.content:
                if item.type != "text":
                    continue

                payload = json.loads(item.text)

                if (
                    not isinstance(payload, dict)
                    or payload.get("ok") is False
                ):
                    raise RuntimeError(
                        "MCP did not confirm ticket creation"
                    )

                record = payload.get("ticket") or payload

                if not isinstance(record, dict):
                    raise RuntimeError(
                        "Unexpected ticket response"
                    )

                ticket_id = record.get("ticket_id")

                if (
                    isinstance(ticket_id, str)
                    and re.fullmatch(r"IT-\d{4}", ticket_id)
                ):
                    return ticket_id

    raise RuntimeError(
        "Ticket ID not confirmed by MCP"
    )


# Health endpoint
@app.get("/health")
def health():
    return {"ok": True}


# Step 1: Generate a ticket draft, without creating it
@app.post("/api/ticket/draft")
async def draft_ticket(request: DraftRequest):

    email = request.email.strip().lower()
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question is required"
        )

    # Run the existing LangGraph Agent
    state = await graph.ainvoke({
        "email": email,
        "question": question,
        "tool_calls": []
    })

    policy = state.get("policy", {})

    # Do not continue without verified employee information
    if (
        state.get("lookup_error")
        or policy.get("needs_clarification")
    ):
        return {
            "reply": state.get("answer"),
            "pending_action": None,
            "tool_calls": state.get("tool_calls", [])
        }

    employee = state.get("employee")
    department = state.get("department")

    if not isinstance(employee, dict) or not department:
        raise HTTPException(
            status_code=422,
            detail="Employee verification failed"
        )

    # A ticket may only be proposed when policy supports it
    suggestion = policy.get("proposed_ticket")

    if not suggestion:
        return {
            "reply": state.get("answer"),
            "pending_action": None,
            "reason": "No ticket proposed by policy",
            "tool_calls": state.get("tool_calls", [])
        }

    # Do not draft when service status is unavailable
    if "error" in state.get("service_status", {}):
        raise HTTPException(
            status_code=503,
            detail="Service status could not be checked"
        )

    # Avoid creating a ticket while incident impact is unclear
    if state.get("incident"):
        return {
            "reply": state.get("answer"),
            "pending_action": None,
            "reason": (
                "Possible active incident; "
                "gateway impact needs verification"
            ),
            "tool_calls": state.get("tool_calls", [])
        }

    # First implementation: Finance VPN escalation
    is_vpn = re.search(
        r"\b(?:vpn|nx-\d+)\b",
        question,
        re.IGNORECASE
    )

    if department != "Finance" or not is_vpn:
        return {
            "reply": state.get("answer"),
            "pending_action": None,
            "reason": (
                "Ticket drafting for this scenario "
                "is not implemented yet"
            ),
            "tool_calls": state.get("tool_calls", [])
        }

    # Extract the error code, if present
    
    # Extract the error code, if present
    code_match = re.search(
        r"\bNX-\d+\b",
        question,
        re.IGNORECASE
    )

    code = (
        code_match.group(0).upper()
        if code_match
        else "access issue"
    )

    # Apply department-specific escalation rules
    if department == "Finance":
        category = "security"
        priority = (
            "P1" if request.affected_users > 3 else "P2"
        )
        kb_refs = ["KB-016", "KB-002"]

        handling = (
            "Finance VPN issues require direct escalation "
            "to Security Operations without client troubleshooting."
        )

    else:
        # Engineering: standard VPN escalation
        category = "network_access"
        priority = (
            "P1" if request.affected_users > 3 else "P3"
        )
        kb_refs = ["KB-002", "KB-015", "KB-016"]

        handling = (
            "Engineering VPN issues follow standard "
            "troubleshooting and escalation rules."
        )

    ticket = {
        "subject": (
            f"{department} VPN {code} - access problem"
        ),
        "description": (
            f"Employee report: {question}\n"
            f"Verified department: {department}.\n"
            f"{handling}\n"
            "The Helpdesk Agent checked the service status."
        ),
        "category": category,
        "priority": priority,
        "requester_email": email,
        "department": department,
        "attempted_steps": request.attempted_steps,
        "kb_refs": kb_refs,
        "affected_users": request.affected_users
    }


    # Build the ticket using official schema fields
    ticket = {
        "subject": f"Finance VPN {code} - access problem",
        "description": (
            f"Employee report: {question}\n"
            "Verified department: Finance.\n"
            "Finance VPN problems must be handled "
            "by Security Operations without client "
            "troubleshooting.\n"
            "The Helpdesk Agent checked the service status."
        ),
        "category": "security",
        "priority": priority,
        "requester_email": email,
        "department": department,
        "attempted_steps": request.attempted_steps,
        "kb_refs": ["KB-016", "KB-002"],
        "affected_users": request.affected_users
    }

    # Verify every field before showing the draft
    validate_ticket(ticket)

    # Create a unique action ID
    action_id = str(uuid.uuid4())

    # Save the draft locally, without calling create_ticket
    pending_actions[action_id] = {
        "status": "pending",
        "ticket": ticket,
        "ticket_id": None
    }

    return {
        "reply": state.get("answer"),
        "pending_action": {
            "action_id": action_id,
            "type": "create_ticket",
            "ticket": ticket
        },
        "tool_calls": state.get("tool_calls", [])
    }


# Step 2: Create the ticket only after explicit approval
@app.post("/api/approve")
async def approve_ticket(request: ActionRequest):

    async with approval_lock:
        record = pending_actions.get(request.action_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Unknown action_id"
            )

        if record["status"] == "rejected":
            raise HTTPException(
                status_code=409,
                detail="Ticket draft was rejected"
            )

        # Return the existing ticket if already approved
        if record["status"] == "approved":
            return {
                "ok": True,
                "ticket_id": record["ticket_id"],
                "reused": True
            }

        if record["status"] != "pending":
            raise HTTPException(
                status_code=409,
                detail="Action cannot be approved"
            )

        try:
            # This is the ONLY place that creates a ticket
            ticket_id = await create_ticket_after_approval(
                record["ticket"],
                request.action_id
            )

        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ticket creation not confirmed: "
                    f"{exc}"
                )
            ) from exc

        record["status"] = "approved"
        record["ticket_id"] = ticket_id

        return {
            "ok": True,
            "ticket_id": ticket_id,
            "reused": False
        }


# Step 3: Reject a pending draft without creating a ticket
@app.post("/api/reject")
async def reject_ticket(request: ActionRequest):

    async with approval_lock:
        record = pending_actions.get(request.action_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Unknown action_id"
            )

        if record["status"] == "approved":
            raise HTTPException(
                status_code=409,
                detail="Ticket already created"
            )

        record["status"] = "rejected"

        return {
            "ok": True,
            "status": "rejected"
        }

