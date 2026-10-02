"""ExecSafe Flask web application for the priority interview."""

from __future__ import annotations

import os
import secrets
import sqlite3
from datetime import datetime, timezone
import json
from typing import Any

from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv

from agent import CATEGORIES, OPEN_ANSWER_CHOICES, REASONING_FEATURES, STAGE_QUESTIONS, extract_reasoning, final_analysis, generated_follow_up, next_question

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
TEST_WITHOUT_LLM = os.getenv("TEST_WITHOUT_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
MAX_STAGES = len(STAGE_QUESTIONS)
INTERVIEWS: dict[str, dict[str, Any]] = {}
DATABASE_PATH = os.getenv("AUDIT_DB_PATH", "audit.db")
JSON_PATH = os.getenv("INTERVIEW_JSON_PATH", "interviews.json")


def _follow_up(stage_record: dict[str, Any]) -> dict[str, str]:
    if TEST_WITHOUT_LLM:
        return {"next_question": "What consequence of your proposed approach would you monitor most closely?"}
    return generated_follow_up(stage_record)


def _reasoning_extraction(stage_record: dict[str, Any]) -> dict[str, Any]:
    if TEST_WITHOUT_LLM:
        return {
            feature: (0 if feature in REASONING_FEATURES[:4] else None)
            for feature in REASONING_FEATURES
        }
    return extract_reasoning(stage_record)


def _analysis(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    if TEST_WITHOUT_LLM:
        scores = {category: 0.0 for category in CATEGORIES}
        scores["Company Safety Merit"] = 1.0
        return {
            "analysis": "Test-mode analysis generated without an LLM.",
            "reasoning_extraction": "Test-mode reasoning summary generated without an LLM.",
            "category": "Company Safety Merit",
            "confidence": "low",
            "scores": scores,
            "evidence": ["The interview completed in offline test mode."],
        }
    return final_analysis(transcript)


def _init_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS interview_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                transcript_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                analysis TEXT NOT NULL,
                final_category TEXT NOT NULL,
                confidence TEXT NOT NULL,
                completed_at TEXT NOT NULL
            )"""
        )


def _save_audit(username: str, transcript: list[dict[str, Any]], result: dict[str, Any]) -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """INSERT INTO interview_audits (
                username, transcript_json, scores_json, analysis,
                final_category, confidence, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                username,
                json.dumps(transcript, ensure_ascii=True),
                json.dumps(result["scores"], ensure_ascii=True),
                result["analysis"],
                result["category"],
                result["confidence"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _save_json(username: str, transcript: list[dict[str, Any]], result: dict[str, Any]) -> None:
    record = {
        "username": username,
        "stages": transcript,
        "reasoning_extraction": result.get("reasoning_extraction", ""),
        "analysis": result,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    records: list[dict[str, Any]] = []
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, encoding="utf-8") as json_file:
                loaded = json.load(json_file)
                if isinstance(loaded, list):
                    records = loaded
        except (OSError, json.JSONDecodeError):
            records = []
    records.append(record)
    with open(JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(records, json_file, ensure_ascii=True, indent=2)


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "completed_at": row["completed_at"],
        "analysis": row["analysis"],
        "category": row["final_category"],
        "confidence": row["confidence"],
        "scores": json.loads(row["scores_json"]),
        "transcript": json.loads(row["transcript_json"]),
    }


def _load_records() -> list[dict[str, Any]]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, username, completed_at, analysis, final_category, confidence, scores_json, transcript_json "
            "FROM interview_audits ORDER BY completed_at DESC, id DESC"
        ).fetchall()
    return [_record_from_row(row) for row in rows]


def _load_record(record_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, username, completed_at, analysis, final_category, confidence, scores_json, transcript_json "
            "FROM interview_audits WHERE id = ?",
            (record_id,),
        ).fetchone()
    return _record_from_row(row) if row else None


_init_database()


def _state() -> dict[str, Any]:
    interview_id = session.get("interview_id")
    if not interview_id or interview_id not in INTERVIEWS:
        interview_id = secrets.token_urlsafe(24)
        session["interview_id"] = interview_id
        INTERVIEWS[interview_id] = {"stage": 1, "phase": "choice", "username": "", "transcript": []}
    return INTERVIEWS[interview_id]


@app.get("/")
def index():
    state = _state()
    return render_template(
        "index.html",
        initial_question=STAGE_QUESTIONS[0]["question"],
        initial_context=STAGE_QUESTIONS[0]["context"],
        stage_title=STAGE_QUESTIONS[0]["title"],
        stage=state["stage"],
        choices=STAGE_QUESTIONS[0]["choices"],
    )


@app.get("/records")
def records():
    return render_template("records.html", records=_load_records())


@app.get("/records/<int:record_id>")
def record_detail(record_id: int):
    record = _load_record(record_id)
    if record is None:
        return render_template("records.html", records=_load_records(), error="Interview record not found."), 404
    return render_template("record_detail.html", record=record)


@app.post("/api/respond")
def respond():
    payload = request.get_json(silent=True) or {}
    answer = str(payload.get("answer", "")).strip()
    choice = str(payload.get("choice", "")).strip()
    username = str(payload.get("username", "")).strip()
    state = _state()
    if state["stage"] > MAX_STAGES:
        return jsonify(error="This interview is already complete."), 400

    if not state["username"]:
        if not username:
            return jsonify(error="Please enter your name before starting."), 400
        if len(username) > 120:
            return jsonify(error="Name must be 120 characters or fewer."), 400
        state["username"] = username

    current_stage = STAGE_QUESTIONS[state["stage"] - 1]
    if state["phase"] == "choice":
        if choice not in current_stage["choices"]:
            return jsonify(error="Please select one of the available choices."), 400
        state["transcript"].append({"stage": state["stage"], "title": current_stage["title"], "context": current_stage["context"], "fixed_question": current_stage["question"], "choice": choice})
        if (state["stage"], choice) in OPEN_ANSWER_CHOICES:
            state["phase"] = "elaboration"
            session.modified = True
            return jsonify(
                done=False,
                phase="elaboration",
                elaboration_question="Please describe the action you would take or the information you need before deciding.",
                stage=state["stage"],
            )
        result = next_question(state["transcript"], state["stage"])
        state["transcript"][-1]["ai_question"] = result["next_question"]
        state["phase"] = "answer"
        session.modified = True
        return jsonify(done=False, phase="answer", ai_question=result["next_question"], stage=state["stage"])

    if not answer:
        return jsonify(error="Please provide a response before continuing."), 400
    if state["phase"] == "elaboration":
        state["transcript"][-1]["elaboration"] = answer
        result = _follow_up(state["transcript"][-1])
        state["transcript"][-1]["ai_question"] = result["next_question"]
        state["phase"] = "answer"
        session.modified = True
        return jsonify(done=False, phase="answer", ai_question=result["next_question"], stage=state["stage"], elaboration=answer)
    state["transcript"][-1]["answer"] = answer
    state["transcript"][-1]["reasoning_extraction"] = _reasoning_extraction(state["transcript"][-1])
    if state["stage"] == MAX_STAGES:
        result = _analysis(state["transcript"])
        _save_audit(state["username"], state["transcript"], result)
        _save_json(state["username"], state["transcript"], result)
        state["stage"] += 1
        session.modified = True
        return jsonify(done=True, result=result, stage=MAX_STAGES)

    state["stage"] += 1
    state["phase"] = "choice"
    next_stage = STAGE_QUESTIONS[state["stage"] - 1]
    session.modified = True
    return jsonify(done=False, phase="choice", fixed_question=next_stage["question"], context=next_stage["context"], title=next_stage["title"], choices=next_stage["choices"], stage=state["stage"])


@app.post("/api/reset")
def reset():
    interview_id = session.pop("interview_id", None)
    if interview_id:
        INTERVIEWS.pop(interview_id, None)
    return jsonify(question=STAGE_QUESTIONS[0]["question"], choices=STAGE_QUESTIONS[0]["choices"], stage=1)


@app.errorhandler(RuntimeError)
def handle_runtime_error(error):
    return jsonify(error=str(error)), 503


@app.errorhandler(ValueError)
def handle_value_error(error):
    return jsonify(error=str(error)), 502


if __name__ == "__main__":
    app.run(debug=True)
