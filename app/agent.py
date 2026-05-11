"""
LangGraph-based conversational agent for SHL assessment recommendation.

State machine:
  START → GUARD → INTENT_EXTRACT → [CLARIFY | COMPARE | RETRIEVE] → RESPOND
                                          ↑
                                       REFINE (re-enters RETRIEVE with updated intent)

The agent is stateless per the API spec — full history is passed each call.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.catalog import get_item_by_name, validate_url, load_catalog
from app.guards import check_message, INJECTION_REPLY, OFF_TOPIC_REPLY
from app.models import ChatResponse, Message, Recommendation
from app.prompts import (
    SYSTEM_PROMPT,
    INTENT_EXTRACTION_PROMPT,
    COMPARISON_PROMPT,
    REFUSAL_NO_CATALOG_MATCH,
)
from app.retrieval import retriever

log = logging.getLogger(__name__)

# ─── Gemini Client Setup ───────────────────────────────────────────────────────
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-2.5-flash"


import time

def _gemini_generate(prompt: str, system: str = None, temperature: float = 0.3) -> str:
    """Call Gemini and return text response. Retries on rate limit (429)."""
    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=2048,
        system_instruction=system,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                log.warning(f"Gemini rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Gemini rate limit exceeded after retries")



# ─── Intent Dataclass ─────────────────────────────────────────────────────────
@dataclass
class UserIntent:
    role: str | None = None
    seniority: str | None = None
    skills: list[str] = field(default_factory=list)
    test_types_requested: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    comparison_request: list[str] | None = None
    has_enough_context: bool = False
    clarification_turns_used: int = 0


def _extract_intent(messages: list[Message]) -> UserIntent:
    """Use Gemini to extract structured intent from the conversation."""
    conversation_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in messages
    )

    prompt = INTENT_EXTRACTION_PROMPT.format(conversation=conversation_text)

    try:
        raw = _gemini_generate(prompt, temperature=0.1)
        # Extract JSON block if wrapped in markdown
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        data = json.loads(raw)

        return UserIntent(
            role=data.get("role"),
            seniority=data.get("seniority"),
            skills=data.get("skills", []),
            test_types_requested=data.get("test_types_requested", []),
            constraints=data.get("constraints", []),
            comparison_request=data.get("comparison_request"),
            has_enough_context=data.get("has_enough_context", False),
            clarification_turns_used=data.get("clarification_turns_used", 0),
        )
    except Exception as e:
        log.warning(f"Intent extraction failed: {e}, using defaults")
        return UserIntent()


def _count_agent_turns(messages: list[Message]) -> int:
    """Count how many assistant turns have already happened."""
    return sum(1 for m in messages if m.role == "assistant")


def _build_search_query(intent: UserIntent) -> str:
    """Build a rich search query from the extracted intent."""
    parts = []
    if intent.role:
        parts.append(intent.role)
    if intent.seniority:
        parts.append(intent.seniority)
    if intent.skills:
        parts.append(" ".join(intent.skills))
    if intent.constraints:
        parts.append(" ".join(intent.constraints))
    # If no parts, use a generic query
    return " ".join(parts) if parts else "general assessment"


def _handle_comparison(intent: UserIntent, messages: list[Message]) -> ChatResponse:
    """Compare two named assessments using catalog data."""
    names = intent.comparison_request
    if not names or len(names) < 2:
        return ChatResponse(
            reply="Could you name the two assessments you'd like me to compare?",
            recommendations=[],
            end_of_conversation=False,
        )

    item1 = get_item_by_name(names[0])
    item2 = get_item_by_name(names[1])

    if not item1 and not item2:
        return ChatResponse(
            reply=f"I couldn't find '{names[0]}' or '{names[1]}' in the SHL catalog. "
                  "Could you check the exact names?",
            recommendations=[],
            end_of_conversation=False,
        )

    if not item1:
        return ChatResponse(
            reply=f"I couldn't find '{names[0]}' in the SHL catalog. "
                  f"Did you mean something else?",
            recommendations=[],
            end_of_conversation=False,
        )

    if not item2:
        return ChatResponse(
            reply=f"I couldn't find '{names[1]}' in the SHL catalog. "
                  f"Did you mean something else?",
            recommendations=[],
            end_of_conversation=False,
        )

    comparison_prompt = COMPARISON_PROMPT.format(
        name1=item1["name"],
        url1=item1["url"],
        type1=", ".join(item1.get("test_type", [])),
        desc1=item1.get("description", "No description available.")[:400],
        levels1=", ".join(item1.get("job_levels", [])) or "All levels",
        name2=item2["name"],
        url2=item2["url"],
        type2=", ".join(item2.get("test_type", [])),
        desc2=item2.get("description", "No description available.")[:400],
        levels2=", ".join(item2.get("job_levels", [])) or "All levels",
    )

    reply = _gemini_generate(comparison_prompt, system=SYSTEM_PROMPT)

    # Include both as recommendations so URLs are in the response
    recs = []
    for item in [item1, item2]:
        if validate_url(item["url"]):
            types = item.get("test_type", ["K"])
            recs.append(Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=types[0] if types else "K",
            ))

    return ChatResponse(
        reply=reply,
        recommendations=recs,
        end_of_conversation=False,
    )


def _handle_recommend(intent: UserIntent, messages: list[Message]) -> ChatResponse:
    """Retrieve and return assessment recommendations."""
    query = _build_search_query(intent)
    log.info(f"Searching with query: '{query}', type_filter: {intent.test_types_requested}")

    results = retriever.search(
        query=query,
        top_k=10,
        test_type_filter=intent.test_types_requested if intent.test_types_requested else None,
    )

    if not results:
        return ChatResponse(
            reply=REFUSAL_NO_CATALOG_MATCH,
            recommendations=[],
            end_of_conversation=False,
        )

    # Validate all URLs (hallucination guard)
    valid_results = [r for r in results if validate_url(r["url"])]
    if not valid_results:
        log.error("All retrieved URLs failed validation!")
        return ChatResponse(
            reply=REFUSAL_NO_CATALOG_MATCH,
            recommendations=[],
            end_of_conversation=False,
        )

    # Cap at 10
    valid_results = valid_results[:10]

    # Build recommendations
    recs = []
    for item in valid_results:
        types = item.get("test_type", ["K"])
        recs.append(Recommendation(
            name=item["name"],
            url=item["url"],
            test_type=types[0] if types else "K",
        ))

    # Generate a natural reply using Gemini
    results_summary = "\n".join(
        f"- {r.name} ({r.test_type}): {r.url}" for r in recs
    )

    role_desc = f"{intent.seniority or ''} {intent.role or 'professional'}".strip()
    skills_desc = f" with skills in {', '.join(intent.skills)}" if intent.skills else ""
    constraints_desc = f" ({', '.join(intent.constraints)})" if intent.constraints else ""

    reply_prompt = (
        f"You are recommending SHL assessments for: {role_desc}{skills_desc}{constraints_desc}.\n\n"
        f"The following {len(recs)} assessments were found:\n{results_summary}\n\n"
        f"Write a brief, natural 2-3 sentence intro explaining why these assessments fit the role. "
        f"Be specific about the match. Don't list the assessments again — they're shown separately."
    )

    reply = _gemini_generate(reply_prompt, system=SYSTEM_PROMPT)

    # Determine if this ends the conversation
    # End when we've given a recommendation (user can still refine)
    eoc = False

    return ChatResponse(
        reply=reply,
        recommendations=recs,
        end_of_conversation=eoc,
    )


def _handle_clarify(intent: UserIntent, messages: list[Message], agent_turns: int) -> ChatResponse:
    """Ask one focused clarifying question."""
    conversation_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in messages[-6:]  # last 6 messages
    )

    clarify_prompt = (
        f"The user wants SHL assessment recommendations but you need more information.\n\n"
        f"Current known info:\n"
        f"- Role: {intent.role or 'unknown'}\n"
        f"- Seniority: {intent.seniority or 'unknown'}\n"
        f"- Skills: {', '.join(intent.skills) if intent.skills else 'unknown'}\n\n"
        f"Recent conversation:\n{conversation_text}\n\n"
        f"Ask exactly ONE concise clarifying question to get the most important missing piece of information. "
        f"{'Ask about the job role/title.' if not intent.role else ''}"
        f"{'Ask about seniority/experience level.' if intent.role and not intent.seniority else ''}"
        f"{'Ask about specific skills or what the role needs to do.' if intent.role and intent.seniority else ''}"
        f"Be direct and friendly. One sentence max."
    )

    question = _gemini_generate(clarify_prompt, system=SYSTEM_PROMPT)

    return ChatResponse(
        reply=question,
        recommendations=[],
        end_of_conversation=False,
    )


async def run_agent(messages: list[Message]) -> ChatResponse:
    """
    Main agent entry point. Stateless — processes full conversation history each call.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_agent_sync, messages)


def _run_agent_sync(messages: list[Message]) -> ChatResponse:
    """Synchronous agent logic (called from thread pool)."""
    if not messages:
        return ChatResponse(
            reply="Hello! I'm your SHL assessment advisor. What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Ensure retriever is ready
    retriever.initialize()

    # Guard: Check latest user message
    latest_user_msg = None
    for m in reversed(messages):
        if m.role == "user":
            latest_user_msg = m.content
            break

    if latest_user_msg:
        is_safe, reason = check_message(latest_user_msg)
        if not is_safe:
            reply = INJECTION_REPLY if reason == "injection" else OFF_TOPIC_REPLY
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # Extract intent from full conversation
    intent = _extract_intent(messages)
    agent_turns = _count_agent_turns(messages)

    log.info(
        f"Intent: role={intent.role}, seniority={intent.seniority}, "
        f"skills={intent.skills}, types={intent.test_types_requested}, "
        f"compare={intent.comparison_request}, enough={intent.has_enough_context}, "
        f"agent_turns={agent_turns}"
    )

    # Comparison branch
    if intent.comparison_request and len(intent.comparison_request) >= 2:
        return _handle_comparison(intent, messages)

    # Turn cap: force recommendation after 3 agent turns
    force_recommend = agent_turns >= 3

    # Clarify or Recommend
    if intent.has_enough_context or force_recommend:
        return _handle_recommend(intent, messages)
    else:
        return _handle_clarify(intent, messages, agent_turns)
