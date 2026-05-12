import os
import json
import logging
import threading
from flask import Flask, render_template, request, jsonify
from pathlib import Path

# Import core RAG components from Fitness_App
from Fitness_App import initialize_system, DEFAULT_USER, clean_answer, extract_user_stats, process_extractions

# ==============================================================
# Flask Configuration
# ==============================================================
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fitness_app")

# ==============================================================
# AI Engine Initialization (Eager, Thread-Safe)
# ==============================================================
_init_lock = threading.Lock()
_sys_engine = None
CONVERSATION_HISTORY = []

def _boot_engine():
    """Load the AI engine exactly once at startup."""
    global _sys_engine
    with _init_lock:
        if _sys_engine is not None:
            return  # Already loaded
        try:
            log.info("=" * 50)
            log.info("  INITIALIZING FITNESS AI ENGINE (Single Load)")
            log.info("=" * 50)
            _sys_engine = initialize_system()
            log.info("  ENGINE READY!")
        except Exception as e:
            log.error(f"  ENGINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            _sys_engine = None

# Boot immediately on import (only runs once because of use_reloader=False)
_boot_engine()


# ==============================================================
# Routes
# ==============================================================

@app.route("/")
def index():
    """Main page serving the modern UI."""
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    """Progress Analytics page serving native Flask UI."""
    return render_template("dashboard.html")

@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns chronological log data for Chart.js rendering."""
    if not _sys_engine: return jsonify({"error": "System not initialized"}), 500
    
    import sqlite3
    db_path = _sys_engine["db_manager"].db_path
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        logs = conn.execute("SELECT * FROM logs WHERE user_id = ? ORDER BY timestamp ASC", (DEFAULT_USER,)).fetchall()
        
    return jsonify([dict(log) for log in logs])

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Retrieve user profile and latest logs for the dashboard."""
    if not _sys_engine:
        return jsonify({"error": "System not initialized"}), 500
    
    profile, latest_log = _sys_engine["db_manager"].get_latest_stats(DEFAULT_USER)
    
    stats = {
        "name": profile["name"] if profile else "Enthusiast",
        "weight": latest_log["weight_kg"] if latest_log else "None",
        "bf_pct": latest_log["body_fat_pct"] if latest_log else "None",
        "goal": profile["goal"] if profile else "Maintenance",
        "split": latest_log["workout_split"] if latest_log else "Not set",
        "bench_pr": f"{latest_log['pr_bench']}kg" if latest_log and latest_log['pr_bench'] else "Not set",
    }
    return jsonify(stats)


@app.route("/api/ask", methods=["POST"])
def ask():
    """Chat endpoint for the AI Coach."""
    global CONVERSATION_HISTORY
    if not _sys_engine:
        return jsonify({"error": "AI Engine offline"}), 500
    
    data = request.json
    question = data.get("prompt", "").strip()
    
    if not question:
        return jsonify({"error": "No prompt provided"}), 400

    # 1. Background Extraction — only if the message looks like it contains stats
    #    (has numbers like "145 lb", "15%", "5'11"). Skip for pure questions/requests.
    import re as _re
    has_numbers = bool(_re.search(r'\d', question))
    
    if has_numbers:
        try:
            new_data = extract_user_stats(_sys_engine["llm"], question)
            if new_data and any(v is not None for v in new_data.values()):
                process_extractions(_sys_engine["db_manager"], new_data, question, DEFAULT_USER)
                # Refresh the chain with new identity context
                from Fitness_App import build_rag_chain
                _sys_engine["chain"], _ = build_rag_chain(
                    _sys_engine["llm"], _sys_engine["retriever"], _sys_engine["db_manager"]
                )
        except Exception as e:
            log.warning(f"Stats extraction failed (non-fatal): {e}")

    # 1.5 Short-Circuit if this was a PURE stat update (e.g. "my bench is 100")
    # This prevents the AI from generating a workout list when you just wanted to log a number.
    intercept_keywords = ["workout", "split", "plan", "give me", "show me", "help", "routine", "day"]
    is_workout_request = any(k in question.lower() for k in intercept_keywords)
    
    if has_numbers and not is_workout_request:
        # Check if we actually found data during extraction
        if 'new_data' in locals() and new_data and any(v is not None for v in new_data.values()):
            keys_found = [k.replace("_", " ").title() for k, v in new_data.items() if v is not None]
            confirm_msg = f"Got it! I've updated your {', '.join(keys_found)}. Great progress!"
            CONVERSATION_HISTORY.append({"user": question, "coach": confirm_msg})
            if len(CONVERSATION_HISTORY) > 2: CONVERSATION_HISTORY.pop(0)
            return jsonify({"answer": confirm_msg})

    # 2. RAG Answer
    
    if CONVERSATION_HISTORY:
        history_str = "\n".join([f"{msg['user']}\n<|im_end|>\n<|im_start|>assistant\n{msg['coach']}\n<|im_end|>\n<|im_start|>user" for msg in CONVERSATION_HISTORY])
        enhanced_question = f"{history_str}\n{question}"
    else:
        enhanced_question = question
        
    try:
        from Fitness_App import check_conversational_intercept
        
        # 1. Programmatic Circuit Breaker
        intercept = check_conversational_intercept(question)
        if intercept:
            CONVERSATION_HISTORY.append({"user": question, "coach": intercept})
            if len(CONVERSATION_HISTORY) > 2:
                CONVERSATION_HISTORY.pop(0)
            return jsonify({"answer": intercept})
            
        # 2. Pass the context-aware question to the RAG chain
        raw_answer = _sys_engine["chain"].invoke(enhanced_question)
        answer = clean_answer(raw_answer)
        
        # Save to short-term memory (keep last 2 exchanges)
        CONVERSATION_HISTORY.append({"user": question, "coach": answer})
        if len(CONVERSATION_HISTORY) > 2:
            CONVERSATION_HISTORY.pop(0)
            
        return jsonify({"answer": answer})
    except Exception as e:
        log.error(f"RAG chain error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    Path("templates").mkdir(exist_ok=True)
    Path("static/css").mkdir(parents=True, exist_ok=True)
    Path("static/js").mkdir(parents=True, exist_ok=True)
    
    # use_reloader=False is CRITICAL:
    # Without it, Flask spawns a second process to watch for file changes,
    # which loads the LLM twice and crashes ChromaDB.
    app.run(debug=True, port=5000, use_reloader=False)
