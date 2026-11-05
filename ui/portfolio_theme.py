"""Visual-only styling for the existing Nakhla Gradio UI.

Does not change authentication, API calls, MCP, evaluation, or approval behavior.
"""
from pathlib import Path
import gradio as gr


def theme():
    return gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate")


CSS = Path(__file__).with_name("portfolio_theme.css").read_text(encoding="utf-8")

HEADER = """
<div class="nakhla-hero">
  <div class="nakhla-hero-copy">
    <div class="nakhla-eyebrow"><span class="nakhla-pulse"></span> NAKHLA SYSTEMS &nbsp; / &nbsp; IT SERVICE DESK</div>
    <div class="nakhla-hero-title">Support that thinks <em>with you.</em></div>
    <p>AI-assisted troubleshooting, grounded knowledge, MCP ticket insights and human-approved escalation.</p>
    <div class="nakhla-hero-stats">
      <span><strong>20</strong> knowledge docs</span>
      <span><strong>6</strong> MCP tools</span>
      <span><strong>10/10</strong> course evaluations</span>
    </div>
  </div>
  <div class="nakhla-hero-mark" aria-hidden="true"><span>N<span class="nakhla-hero-mark-dot">.</span></span><small>IT / AGENT</small></div>
</div>
<div class="nakhla-context-note">LOCAL DEVELOPMENT DEMO <span>•</span> Answers are grounded in the project's fictional helpdesk knowledge base and simulated system.</div>
"""
