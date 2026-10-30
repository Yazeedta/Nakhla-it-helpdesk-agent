# Development Handoff

Project: Nakhla IT Helpdesk Agent

## Working
- RAG knowledge-base answers and citations
- MCP employee lookup and service status
- LangGraph active-incident workflow
- Ticket drafting and approval gate
- Multi-user demo login
- MCP ticket history and ticket details

## Known issue
Ticket history isolation in the Gradio UI is not fully fixed.
Sara sometimes sees Tariq's IT-1002 in the ticket list,
although direct MCP ownership checks return the correct records.

Expected:
- Tariq: IT-1002
- Sara: IT-1021

Do not expose this demo publicly until the isolation issue is fixed.

## Important
- Never commit .env or demo user credentials.
- Keep EVAL_ENABLE_WRITES=0 for public demonstrations.
- Re-run the 10 evaluation cases after making changes.
- Do not modify the simulated services supplied by the course.
