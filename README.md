# ExecSafe

A Flask decision interview based on the ExecSafe Scenario 01 specification. The participant acts as the Managing Director of a construction company in Singapore and receives Project Alpha information progressively across three stages: Initial Safety Concern, Production Pressure, and Conflicting Safety Information.

Each stage presents the specified fixed forced-choice scenario and captures the participant's reasoning. The open-answer choices (Stage 1-D, Stage 2-C, and Stage 3-E) first ask the participant to elaborate, then use OpenAI to generate a focused follow-up question. Other choices use the predefined follow-up questions. There is no predetermined correct response.

## Setup

1. Create and activate a virtual environment if desired.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and a strong `FLASK_SECRET_KEY`.
4. Run: `python app.py`
5. Open `http://127.0.0.1:5000`.

For an offline test run, set `TEST_WITHOUT_LLM=true`. The app then uses deterministic follow-up questions, reasoning extraction, and final analysis without contacting OpenAI.

After each participant response, GenAI reasoning extraction codes observable features: safety consideration, production consideration, evidence seeking, uncertainty acknowledgement, information source, action scope, and emergent consideration. These are not treated as validated scores or judgements of executive safety leadership. The final step returns the classification, confidence, comparative category scores, evidence, and an overall reasoning summary.

Completed interviews are appended to `interviews.json`. Set `INTERVIEW_JSON_PATH` to choose another location.

## Audit database

Completed interviews are saved to `audit.db` in the `interview_audits` table. Each row contains the username, full question-and-response transcript, category scores, analysis, selected category, confidence, and UTC completion timestamp. Set `AUDIT_DB_PATH` to use another SQLite file. The database is ignored by Git because it contains personal responses.
# ExecSafe
