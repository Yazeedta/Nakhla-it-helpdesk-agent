
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_graph import graph
from ticket_gate import (
    DraftRequest,
    approval_lock,
    create_ticket_after_approval,
    draft_ticket,
    pending_actions,
)


# FastAPI application used by the Gradio UI
app = FastAPI(title="Nakhla IT Helpdesk Agent")

# Temporary conversation memory
sessions = {}


# Request models
class ChatUser(BaseModel):
    email: str | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    user: ChatUser | None = None


class ApprovalRequest(BaseModel):
    session_id: str
    action_id: str
    approved: bool


# Convert LangGraph tool logs to the UI format
def format_calls(calls):
    return [
        {
            "tool": item.get("name", item.get("tool", "unknown")),
            "ok": item.get("ok", True),
            "ms": item.get("ms", "-")
        }
        for item in (calls or [])
        if isinstance(item, dict)
    ]


# Extract citations actually used in the answer
def sources_from_state(state):
    answer = state.get("answer") or ""

    cited = set(
        re.findall(r"\[((?:KB|SP)-\d+)\]", answer)
    )

    found = {}

    for chunk in state.get("evidence") or []:
        doc_id = chunk["document_id"]

        if doc_id in cited:
            found[doc_id] = {
                "id": doc_id,
                "title": chunk["metadata"].get("title", "")
            }

    return list(found.values())


# Build a standard UI response
def response(
    reply,
    *,
    pending=None,
    calls=None,
    citations=None,
    state=None,
    result=None
):
    return {
        "reply": reply,
        "pending_action": pending,
        "citations": citations or [],
        "tool_calls": format_calls(calls),
        "state": state,
        "result": result or {}
    }


# Detect requests to open a helpdesk ticket
def ticket_requested(message):
    text = message.lower()

    return bool(
        re.search(
            r"\b(?:create|open|file|raise|submit|want|need|make)"
            r"\b.{0,50}\bticket\b",
            text
        )
        or re.search(r"\bescalat(?:e|ion)\b", text)
        or re.search(
            r"(?:افتح|بدي|اعمل).{0,30}(?:تذكرة|تيكت)",
            text
        )
        or "تصعيد" in text
    )


# Convert an existing ticket draft to the UI card format
def pending_card(record, action_id):
    ticket = record["ticket"]

    return {
        "action_id": action_id,
        "type": "create_ticket",
        "summary": ticket["subject"],
        "payload": ticket
    }


# Health endpoint
@app.get("/health")
def health():
    return {"ok": True}


# Main chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        return response("Please describe your IT problem.")

    # Get or create the conversation session
    session = sessions.setdefault(
        request.session_id,
        {
            "email": None,
            "last_question": None,
            "awaiting_steps": False,
            "pending_action_id": None,
            "offered_escalation": False
        }
    )

    # Read the employee email from the UI
    email = (
        (request.user.email or "")
        if request.user
        else ""
    ).strip().lower()

    if email:
        if session["email"] and session["email"] != email:
            return response(
                "This session belongs to a different work email. "
                "Start a new session."
            )

        session["email"] = email

    if not session["email"]:
        return response(
            "Please enter your company email address above."
        )

    # Do not create another draft while approval is pending
    pending_id = session.get("pending_action_id")

    if pending_id:
        record = pending_actions.get(pending_id)

        if record and record["status"] == "pending":
            return response(
                "A draft is already awaiting your "
                "approval or rejection.",
                pending=pending_card(record, pending_id),
                state="pending_approval"
            )

    # Recognize a confirmation to a previous escalation offer
    confirmed_yes = message.lower() in {
        "yes",
        "yes please",
        "please do",
        "نعم",
        "اه",
        "آه",
        "موافق"
    }

    wants_ticket = (
        ticket_requested(message)
        or (
            confirmed_yes
            and session["offered_escalation"]
        )
    )

    # Ask for attempted steps before preparing a ticket
    if wants_ticket and not session["awaiting_steps"]:
        if not session["last_question"]:
            return response(
                "Please describe the IT problem first, "
                "then I can prepare a ticket draft."
            )

        session["awaiting_steps"] = True

        return response(
            "What troubleshooting steps have you already tried? "
            "If you haven't tried anything, reply 'none'. "
            "I will prepare a draft, not submit a ticket.",
            state="awaiting_attempted_steps"
        )

    # The employee is answering the attempted-steps question
    if session["awaiting_steps"]:
        steps = message

        if steps.lower() in {
            "none",
            "nothing",
            "no steps",
            "ولا اشي",
            "ولا شيء"
        }:
            steps = (
                "No troubleshooting steps attempted "
                "(confirmed by requester)."
            )

        if not 3 <= len(steps) <= 300:
            return response(
                "Please describe the attempted steps "
                "in 3-300 characters, or type 'none'."
            )

        session["awaiting_steps"] = False

        # Reuse our existing ticket draft logic
        try:
            draft = await draft_ticket(
                DraftRequest(
                    email=session["email"],
                    question=session["last_question"],
                    attempted_steps=[steps],
                    affected_users=1
                )
            )

        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to prepare ticket draft: {exc}"
            ) from exc

        pending = draft.get("pending_action")

        if not pending:
            return response(
                (draft.get("reply") or "")
                + "\n\n"
                + draft.get(
                    "reason",
                    "Ticket draft unavailable."
                ),
                calls=draft.get("tool_calls"),
                state="answered"
            )

        action_id = pending["action_id"]
        record = pending_actions[action_id]

        session["pending_action_id"] = action_id

        return response(
            "Ticket draft prepared. "
            "Review the details and choose Approve or Reject. "
            "No ticket has been created yet.",
            pending=pending_card(record, action_id),
            calls=draft.get("tool_calls"),
            state="pending_approval"
        )

    # Normal chat question: run our existing LangGraph Agent
    session["last_question"] = message

    try:
        state = await graph.ainvoke(
            {
                "question": message,
                "email": session["email"],
                "tool_calls": []
            }
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Agent is unavailable: {exc}"
        ) from exc

    answer = (
        state.get("answer")
        or "I couldn't verify an answer."
    )

    session["offered_escalation"] = bool(
        re.search(
            r"\b(?:ticket|escalat\w*)\b",
            answer,
            re.IGNORECASE
        )
    )

    return response(
        answer,
        calls=state.get("tool_calls"),
        citations=sources_from_state(state),
        state="answered"
    )


# Approve or reject from the UI
@app.post("/api/approve")
async def approve(request: ApprovalRequest):
    session = sessions.get(request.session_id)

    if (
        not session
        or session.get("pending_action_id")
        != request.action_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Action does not belong to this session"
        )

    async with approval_lock:
        record = pending_actions.get(request.action_id)

        if not record:
            raise HTTPException(
                status_code=404,
                detail="Unknown or expired draft"
            )

        if (
            record["ticket"]["requester_email"]
            != session["email"]
        ):
            raise HTTPException(
                status_code=403,
                detail="Requester does not match this session"
            )

        # Reject without calling create_ticket
        if not request.approved:
            if record["status"] == "approved":
                raise HTTPException(
                    status_code=409,
                    detail="Ticket was already created"
                )

            record["status"] = "rejected"

            return response(
                "Ticket draft rejected. No ticket was created.",
                state="rejected",
                result={"status": "rejected"}
            )

        # Prevent approval after rejection
        if record["status"] == "rejected":
            raise HTTPException(
                status_code=409,
                detail="Ticket draft was rejected"
            )

        # Reuse the existing ticket after repeated approval
        if record["status"] == "approved":
            return response(
                f"Ticket {record['ticket_id']} "
                "was already created.",
                state="approved",
                result={
                    "ticket_id": record["ticket_id"],
                    "reused": True
                }
            )

        # Only here do we create a real ticket via MCP
        try:
            ticket_id = await create_ticket_after_approval(
                record["ticket"],
                request.action_id
            )

        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Ticket creation not confirmed: {exc}"
            ) from exc

        record["status"] = "approved"
        record["ticket_id"] = ticket_id

        return response(
            f"Ticket {ticket_id} created successfully.",
            state="approved",
            calls=[
                {"name": "create_ticket", "ok": True}
            ],
            result={
                "ticket_id": ticket_id,
                "reused": False
            }
        )

