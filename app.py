from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from judge.problems import (
    get_problem_feedback,
    get_problem_public,
    get_problem_solution,
    list_problem_briefs,
    save_problem_feedback,
)
from judge.runner import run_problem


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/problems")
def api_problems():
    return jsonify({"problems": list_problem_briefs()})


@app.get("/api/problems/<problem_id>")
def api_problem_detail(problem_id: str):
    problem = get_problem_public(problem_id)
    if problem is None:
        return jsonify({"ok": False, "error": f"Unknown problem_id: {problem_id}"}), 404
    return jsonify({"ok": True, "problem": problem})


@app.get("/api/problems/<problem_id>/solution")
def api_problem_solution(problem_id: str):
    solution = get_problem_solution(problem_id)
    if solution is None:
        return jsonify({"ok": False, "error": f"Unknown problem_id: {problem_id}"}), 404
    return jsonify({"ok": True, "solution": solution})


@app.get("/api/problems/<problem_id>/feedback")
def api_problem_feedback(problem_id: str):
    feedback = get_problem_feedback(problem_id)
    if feedback is None:
        return jsonify({"ok": False, "error": f"Unknown problem_id: {problem_id}"}), 404
    return jsonify({"ok": True, "feedback": feedback})


@app.post("/api/problems/<problem_id>/feedback")
def api_problem_feedback_save(problem_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        feedback = save_problem_feedback(problem_id, payload)
        problem = get_problem_public(problem_id)
        solution = get_problem_solution(problem_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify(
        {
            "ok": True,
            "feedback": feedback,
            "problem": problem,
            "solution": solution,
        }
    )


@app.post("/api/test")
def api_test():
    payload = request.get_json(silent=True) or {}
    problem_id = str(payload.get("problem_id", "")).strip()
    code = str(payload.get("code", ""))
    try:
        custom_cases = _parse_custom_cases(payload.get("custom_cases"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if not problem_id:
        return jsonify({"ok": False, "error": "problem_id is required"}), 400
    if not code.strip():
        return jsonify({"ok": False, "error": "code is required"}), 400

    result = run_problem(problem_id, code, mode="test", custom_cases=custom_cases)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/submit")
def api_submit():
    payload = request.get_json(silent=True) or {}
    problem_id = str(payload.get("problem_id", "")).strip()
    code = str(payload.get("code", ""))

    if not problem_id:
        return jsonify({"ok": False, "error": "problem_id is required"}), 400
    if not code.strip():
        return jsonify({"ok": False, "error": "code is required"}), 400

    result = run_problem(problem_id, code, mode="submit", custom_cases=None)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


def _parse_custom_cases(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        parsed = json.loads(stripped)
        if parsed is None:
            return None
        if not isinstance(parsed, list):
            raise ValueError("custom_cases must be a JSON array")
        return parsed
    raise ValueError("custom_cases must be null, array, or JSON string")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)
