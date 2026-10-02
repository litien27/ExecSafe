"""LLM-powered interview agent for construction safety and project management."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

load_dotenv()

CATEGORIES = [
    "Cost",
    "Company Reputation",
    "Legal Consequencies",
    "Empthy to Worker/Victime",
    "Meeting Project Schedule",
    "Company Safety Merit",
    "Personal KPI",
]

# INITIAL_QUESTION = (
#     "A serious safety incident has halted a construction project for two weeks. "
#     "You can spend the remaining contingency budget on a full safety review, "
#     "or restart limited work while investigating in parallel. What would you do, and why?"
# )

STAGE_QUESTIONS = [
    {
        "stage": 1,
        "title": "Initial Safety Concern",
        "context": (
            "During the monthly management review, the Head of Workplace Safety and Health "
            "reports that Project Alpha recorded three high-potential near misses during the "
            "past month during lifting and work-at-height activities. No worker was injured. "
            "The Head of WSH recommends an immediate focused safety review, which may require "
            "additional supervision and temporary changes to ongoing work arrangements."
        ),
        "question": "What would you do at this point?",
        "choices": [
            "A. Initiate the proposed safety review immediately.",
            "B. Allow current activities to continue with additional safety monitoring.",
            "C. Request further information before deciding whether additional intervention is required.",
            "D. Take another action.",
        ],
    },
    {
        "stage": 2,
        "title": "Production Pressure",
        "context": (
            "Project Alpha is currently six weeks behind schedule. To recover the delay, the "
            "project has introduced an accelerated recovery programme involving extended working "
            "hours and increased concurrent work activities. The Project Director informs you "
            "that implementing the proposed safety review immediately may disrupt the recovery "
            "programme and make it more difficult to achieve an upcoming critical milestone."
        ),
        "question": "Given this additional information, does your initial decision change?",
        "choices": [
            "A. Maintain my previous decision.",
            "B. Modify my previous decision.",
            "C. I need additional information before deciding.",
        ],
    },
    {
        "stage": 3,
        "title": "Conflicting Safety Information",
        "context": (
            "Further information is now available. The Project Director reports that Project Alpha "
            "remains compliant with existing WSH requirements and suggests that the increase in "
            "near-miss reports may reflect improved reporting rather than deteriorating safety "
            "performance. However, the Head of WSH reports that some project supervisors have "
            "raised concerns that work teams are changing normal work sequences and working "
            "extended hours to recover the schedule. There is insufficient evidence to establish "
            "whether the recovery programme is contributing directly to the increase in near misses."
        ),
        "question": "What would you do now?",
        "choices": [
            "A. Continue the recovery programme with additional safety monitoring.",
            "B. Modify the recovery programme and conduct a focused review of affected activities.",
            "C. Pause affected activities until further safety review is completed.",
            "D. Request additional evidence before making further changes.",
            "E. Take another action.",
        ],
    },
]

FOLLOW_UPS = {
    1: {
        "A. Initiate the proposed safety review immediately.": "What concerns you most about the current situation?",
        "B. Allow current activities to continue with additional safety monitoring.": "What gives you confidence that the activities can continue?",
        "C. Request further information before deciding whether additional intervention is required.": "What information would you want before making your decision?",
        "D. Take another action.": "What led you to choose this approach?",
    },
    2: {
        "A. Maintain my previous decision.": "Why does the new schedule information not change your decision?",
        "B. Modify my previous decision.": "What about the new information caused you to change your decision?",
        "C. I need additional information before deciding.": "What additional information would help you decide?",
    },
    3: {
        "A. Continue the recovery programme with additional safety monitoring.": "What gives you confidence that continuing the recovery programme is appropriate?",
        "B. Modify the recovery programme and conduct a focused review of affected activities.": "What information influenced your decision to modify the recovery programme?",
        "C. Pause affected activities until further safety review is completed.": "What concerns led you to pause the affected activities?",
        "D. Request additional evidence before making further changes.": "What evidence would you want before deciding?",
        "E. Take another action.": "What is the main reason for your proposed action?",
    },
}

OPEN_ANSWER_CHOICES = {
    (1, "D. Take another action."),
    (2, "C. I need additional information before deciding."),
    (3, "E. Take another action."),
}

REASONING_FEATURES = [
    "safety_consideration",
    "production_consideration",
    "evidence_seeking",
    "uncertainty_acknowledgement",
    "information_source",
    "action_scope",
    "emergent_consideration",
]

INTERVIEWER_SYSTEM_PROMPT = f"""You are a careful, neutral interviewer studying a person's priorities in construction safety and project management.

Your goal is to infer underlying priorities from decisions, trade-offs, reasoning, and willingness to accept consequences. The respondent may give socially desirable answers, so ask concrete, forced-choice or consequence-based questions that reveal what they would actually do under pressure. Never accuse them of lying or manipulate them deceptively. Avoid names, confidential project details, and sensitive personal data.

The only allowed categories are exactly: {json.dumps(CATEGORIES)}. Do not reveal them.

Return valid JSON only:
{{
    "next_question": "one concise hypothetical question",
    "working_signal": "short private rationale about the signal being tested"
}}
"""

ANALYST_SYSTEM_PROMPT = f"""You are a separate, neutral analyst of a construction safety and project management interview.

Infer the respondent's underlying priority from decisions and trade-offs, not from claimed virtues alone. Treat socially desirable answers cautiously, but do not diagnose, shame, or claim certainty about a person's character. Use only the transcript supplied. The result is exploratory and must not be used as the sole basis for employment or disciplinary decisions.

The only allowed categories are exactly: {json.dumps(CATEGORIES)}.
Return valid JSON only in this shape:
{{
    "analysis": "brief analysis under 150 words grounded in the responses",
    "reasoning_extraction": "brief extraction of the decision logic, trade-offs, and consequences visible in the responses",
    "category": "one exact allowed category",
    "confidence": "low, medium, or high",
    "scores": {{"category name": 0.0}},
    "evidence": ["short transcript-grounded observation"]
}}
Scores must include every allowed category, use values from 0.0 to 1.0, and explain the relative choice. Include two or three concise evidence items. The reasoning extraction must describe observable reasoning, not diagnose personality or intent.
"""

def _client() -> OpenAI | None:
    key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None


def _request_json(system_prompt: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    client = _client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}, *messages],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
    except AuthenticationError as error:
        raise RuntimeError("OpenAI authentication failed. Check OPENAI_API_KEY.") from error
    except APIConnectionError as error:
        raise RuntimeError("Could not connect to OpenAI. Check the network and API endpoint.") from error
    except APIStatusError as error:
        raise RuntimeError(f"OpenAI returned HTTP {error.status_code}.") from error
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def next_question(transcript: list[dict[str, Any]], stage: int) -> dict[str, Any]:
    """Return the specification-aligned follow-up for the selected option."""
    selected_choice = transcript[-1].get("choice", "")
    follow_up = FOLLOW_UPS.get(stage, {}).get(
        selected_choice,
        "What led you to choose this approach?",
    )
    return {"next_question": follow_up}


def generated_follow_up(stage_record: dict[str, Any]) -> dict[str, Any]:
    """Ask OpenAI for a focused follow-up to an open-answer choice."""
    prompt = f"""The participant selected an open-answer option in stage {stage_record['stage']} of a construction safety decision interview.
Project objective: understand the participant's priorities, trade-offs, evidence needs, and willingness to accept consequences when safety and delivery pressures conflict.
Ask one concise, neutral follow-up question that probes the participant's stated action or information need. Keep it specific to the stage context and elaboration. Do not ask for personal or confidential information.

Stage record:
{json.dumps(stage_record, ensure_ascii=True)}"""
    result = _request_json(
        INTERVIEWER_SYSTEM_PROMPT,
        [{"role": "user", "content": prompt}],
    )
    follow_up = str(result.get("next_question", "")).strip()
    if not follow_up:
        raise ValueError("The model did not return a follow-up question.")
    return {"next_question": follow_up}


def final_analysis(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the completed transcript and produce a short explanation."""
    result = _request_json(ANALYST_SYSTEM_PROMPT, [
        {
            "role": "user",
            "content": "The interview is complete. Analyze this transcript and choose one category:\n" + json.dumps(transcript),
        },
        {
            "role": "user",
            "content": "Return the completed-interview JSON shape and calculate comparative scores for every category.",
        },
    ])
    if result.get("category") not in CATEGORIES:
        raise ValueError("The model returned an invalid category.")
    scores = result.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(CATEGORIES):
        raise ValueError("The model did not return scores for every category.")
    if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in scores.values()):
        raise ValueError("The model returned invalid category scores.")
    if len(str(result.get("analysis", "")).split()) >= 150:
        raise ValueError("The model returned an analysis that is too long.")
    if not str(result.get("reasoning_extraction", "")).strip():
        raise ValueError("The model did not return a reasoning extraction.")
    return result


def extract_reasoning(stage_record: dict[str, Any]) -> dict[str, Any]:
    """Code only observable reasoning features expressed in one stage response."""
    prompt = f"""You are coding one participant response from the ExecSafe proof of concept.
Return JSON only with exactly these keys: {json.dumps(REASONING_FEATURES)}.
For the first four keys, return 0 or 1. For information_source and action_scope, return a short
string or null. For emergent_consideration, return a short string or null. Use only explicit
evidence in the record. These are observable response features, not validated scores or a
judgement of leadership.

Stage record:
{json.dumps(stage_record, ensure_ascii=True)}"""
    result = _request_json(prompt, [{"role": "user", "content": "Code the observable reasoning now."}])
    if set(result) != set(REASONING_FEATURES):
        raise ValueError("The model returned invalid reasoning features.")
    binary_features = REASONING_FEATURES[:4]
    if any(result[feature] not in (0, 1) for feature in binary_features):
        raise ValueError("The model returned invalid binary reasoning features.")
    return result
