"""
Fitness RAG Application
=======================
A Retrieval-Augmented Generation pipeline for fitness data.
Uses a pre-built ChromaDB vector store (built by vectorize.py).

Usage:
    python Fitness_App.py
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

# -- LangChain --
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# -- Data/ML --
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# ==============================================================
# Configuration
# ==============================================================
DATA_DIR = Path(__file__).parent / "dataset"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
TOP_K = 15
CSV_ROWS_PER_DOC = 100
DB_PATH = Path(__file__).parent / "fitness_app.db"
DEFAULT_USER = "default_user"

# ==============================================================
# Logging
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ==============================================================
# User Database Manager
# ==============================================================

class UserProfileManager:
    def __init__(self, db_path):
        import sqlite3
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    gender TEXT,
                    birthdate TEXT,
                    height_cm REAL,
                    activity_level TEXT DEFAULT 'moderate',
                    goal TEXT DEFAULT 'maintenance'
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    weight_kg REAL,
                    body_fat_pct REAL,
                    abs_appearance TEXT,
                    pr_bench REAL,
                    bench_reps INTEGER,
                    pr_squat REAL,
                    squat_reps INTEGER,
                    pr_deadlift REAL,
                    deadlift_reps INTEGER,
                    workout_split TEXT,
                    FOREIGN KEY (user_id) REFERENCES profiles (user_id)
                )
            """)
            # Migrations
            cursor.execute("PRAGMA table_info(logs)")
            cols = [c[1] for c in cursor.fetchall()]
            for new_col, col_def in [("bench_reps", "INTEGER"), ("squat_reps", "INTEGER"), ("deadlift_reps", "INTEGER")]:
                if new_col not in cols:
                    cursor.execute(f"ALTER TABLE logs ADD COLUMN {new_col} {col_def}")
            conn.commit()

    def upsert_profile(self, user_id, name=None, gender=None, birthdate=None, height_cm=None, activity_level=None, goal=None):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM profiles WHERE user_id = ?", (user_id,))
            if cursor.fetchone():
                updates, params = [], []
                for k, v in [("name", name), ("gender", gender), ("birthdate", birthdate), ("height_cm", height_cm), ("activity_level", activity_level), ("goal", goal)]:
                    if v is not None:
                        updates.append(f"{k} = ?")
                        params.append(v)
                if updates:
                    params.append(user_id)
                    cursor.execute(f"UPDATE profiles SET {', '.join(updates)} WHERE user_id = ?", params)
            else:
                cursor.execute(
                    "INSERT INTO profiles (user_id, name, gender, birthdate, height_cm, activity_level, goal) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, name or "User", gender or "Not specified", birthdate or "Unknown", height_cm or 0.0, activity_level or "moderate", goal or "maintenance")
                )
            conn.commit()

    def add_log(self, user_id, weight=None, body_fat=None, abs_app=None, bench=None, bench_reps=None, squat=None, squat_reps=None, deadlift=None, deadlift_reps=None, split=None):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch previous log to carry forward any values not being updated
            cursor.execute("SELECT * FROM logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
            prev = cursor.fetchone()
            
            # Merge: use new value if provided, otherwise keep previous
            if prev:
                weight = weight if weight is not None else prev["weight_kg"]
                body_fat = body_fat if body_fat is not None else prev["body_fat_pct"]
                abs_app = abs_app if abs_app is not None else prev["abs_appearance"]
                bench = bench if bench is not None else prev["pr_bench"]
                bench_reps = bench_reps if bench_reps is not None else prev["bench_reps"]
                squat = squat if squat is not None else prev["pr_squat"]
                squat_reps = squat_reps if squat_reps is not None else prev["squat_reps"]
                deadlift = deadlift if deadlift is not None else prev["pr_deadlift"]
                deadlift_reps = deadlift_reps if deadlift_reps is not None else prev["deadlift_reps"]
                split = split if split is not None else prev["workout_split"]
            
            cursor.execute(
                "INSERT INTO logs (user_id, weight_kg, body_fat_pct, abs_appearance, pr_bench, bench_reps, pr_squat, squat_reps, pr_deadlift, deadlift_reps, workout_split) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, weight, body_fat, abs_app, bench, bench_reps, squat, squat_reps, deadlift, deadlift_reps, split)
            )
            conn.commit()

    def get_latest_stats(self, user_id):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,))
            profile = cursor.fetchone()
            cursor.execute("SELECT * FROM logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
            log_entry = cursor.fetchone()
            return profile, log_entry

    def calculate_calories(self, profile, latest_log):
        if not profile or not latest_log or not latest_log["weight_kg"] or not profile["height_cm"]:
            return None, None
        try:
            from datetime import datetime
            birth = datetime.strptime(profile["birthdate"], "%Y-%m-%d")
            age = (datetime.now() - birth).days // 365
        except:
            age = 25
        try:
            weight = float(latest_log["weight_kg"])
            height = float(profile["height_cm"])
        except (TypeError, ValueError):
            return None, None
        is_male = str(profile["gender"]).lower() == "male"
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + (5 if is_male else -161)
        multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very active": 1.9}
        tdee = bmr * multipliers.get(profile["activity_level"].lower(), 1.2)
        return round(bmr), round(tdee)

    def get_context_summary(self, user_id):
        profile, log_entry = self.get_latest_stats(user_id)
        if not profile: return "User Profile: Not yet configured."
        summary = [f"User Profile ({profile['user_id']}):"]
        summary.append(f"- Name: {profile['name']}, Gender: {profile['gender']}, Height: {profile['height_cm']}cm")
        summary.append(f"- Goal: {profile['goal']}, Activity: {profile['activity_level']}")
        if log_entry:
            summary.append(f"- Current Weight: {log_entry['weight_kg']}kg, Body Fat: {log_entry['body_fat_pct']}%")
            summary.append(f"- Abs: {log_entry['abs_appearance']}, Split: {log_entry['workout_split']}")
            prs = []
            if log_entry['pr_bench']: prs.append(f"Bench: {log_entry['pr_bench']}kg x {log_entry['bench_reps'] or 1}")
            if log_entry['pr_squat']: prs.append(f"Squat: {log_entry['pr_squat']}kg x {log_entry['squat_reps'] or 1}")
            if log_entry['pr_deadlift']: prs.append(f"Deadlift: {log_entry['pr_deadlift']}kg x {log_entry['deadlift_reps'] or 1}")
            if prs: summary.append("- PRs: " + ", ".join(prs))
            bmr, tdee = self.calculate_calories(profile, log_entry)
            if bmr: summary.append(f"- BMR: {bmr} kcal, TDEE: {tdee} kcal")
        return "\n".join(summary)

    def get_progress_summary(self, user_id, days=7):
        import sqlite3
        from datetime import datetime, timedelta
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
            current = cursor.fetchone()
            if not current: return "No logs found."
            target_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("SELECT * FROM logs WHERE user_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1", (user_id, target_date))
            past = cursor.fetchone()
            if not past: return f"Current weight: {current['weight_kg']}kg."
            diff_w = round(current['weight_kg'] - past['weight_kg'], 1)
            return f"--- Weekly Progress ---\nWeight: {current['weight_kg']}kg ({'+' if diff_w > 0 else ''}{diff_w}kg)"


# ==============================================================
# Helpers
# ==============================================================

def lbs_to_kg(lbs): return round(float(lbs) * 0.453592, 1)

def parse_us_height(h_str):
    import re
    match = re.search(r"(\d+)['\s]*(?:ft|feet)?\s*(\d+)?", str(h_str).lower())
    if match:
        f, i = match.group(1), match.group(2) or 0
        return round(((float(f) * 12) + float(i)) * 2.54, 1)
    return None

def build_llm():
    log.info(f"Loading LLM: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, quantization_config=bnb_config, device_map="auto")
    
    from transformers import GenerationConfig
    
    # 1. Clear any confusing defaults from the model's official config
    gen_config = GenerationConfig.from_pretrained(LLM_MODEL)
    gen_config.max_new_tokens = 2048
    gen_config.temperature = 0.3
    gen_config.do_sample = True
    gen_config.repetition_penalty = 1.1
    
    # 2. Key: Explicitly remove the deprecated max_length property
    if hasattr(gen_config, "max_length"):
        del gen_config.max_length

    # 3. Apply it to the model. The pipeline below will inherit it automatically.
    model.generation_config = gen_config
    
    # 4. Minimal Pipeline: Passing NO generation parameters here 
    # to avoid the "Passing generation_config together with..." error.
    text_pipeline = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=text_pipeline)

def build_rag_chain(llm, retriever, db_manager, user_id=DEFAULT_USER):
    _IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
    user_context = db_manager.get_context_summary(user_id)
    prompt_template = (
        f"{_IM_START}system\n"
        "**SYSTEM — FITNESS AI COACH**\n\n"
        "You operate strictly in one of these modes depending on the user's input:\n\n"
        "### MODE A: CONVERSATIONAL & Q&A\n"
        "- If the user says a greeting or asks a general fitness question (e.g., 'what is a squat?'), answer them naturally and briefly.\n"
        "- DO NOT generate any workouts or ask about muscle groups.\n\n"
        "### MODE B: SINGLE WORKOUT GENERATOR\n"
        "- If the user asks for a simple, one-off workout (e.g., 'give me a leg workout'), generate EXACTLY one workout using the CONTEXT.\n"
        "- Format it exactly like this:\n"
        "**[Muscle Group] Workout**\n"
        "- Exercise 1 (3 x 10)\n"
        "- Exercise 2 (4 x 8)\n"
        "- Exercise 3 (3 x 12)\n"
        "- STOP immediately after listing. DO NOT mention 'Day 1' and DO NOT ask what to do for the next day.\n\n"
        "### MODE C: MULTI-DAY SPLIT BUILDER\n"
        "ONLY activate this if the user explicitly asks for a 'split' (e.g., '3-day split', '4-day split'):\n"
        "1. Acknowledge the split and ask: **'What muscle group do you want to train on Day 1?'** DO NOT generate any exercises until they answer.\n"
        "2. When they provide the muscles for Day 1, generate Day 1 in this format:\n"
        "**Day X — [Muscle Group]**\n"
        "- Exercise 1 (3 x 10)\n"
        "- Exercise 2 (4 x 8)\n"
        "3. At the very end of your response, you MUST ask: **'What muscle group do you want to train on Day [Next Day]?'**\n"
        "4. Never output multiple days at once. Never skip ahead.\n\n"
        "### GENERAL DETERMINISTIC RULES\n"
        "- Use only the exact muscle groups requested.\n"
        "- Never hallucinate. Answer ONLY using facts and exercises from the CONTEXT.\n"
        "- You MUST replace placeholder sets/reps with actual numbers.\n"
        "- If the user's message is unclear, ask a clarifying question. Never assume.\n"
        f"{_IM_END}\n"
        f"{_IM_START}user\n"
        f"USER PROFILE:\n{user_context}\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n"
        f"{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    # --- HYBRID RETRIEVAL ---
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        from langchain_core.documents import Document
        
        # Extract all chunks from the Chroma DB to build the BM25 keyword index
        vectorstore_data = retriever.vectorstore.get(include=["documents", "metadatas"])
        if vectorstore_data and "documents" in vectorstore_data and len(vectorstore_data["documents"]) > 0:
            bm25_docs = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(vectorstore_data["documents"], vectorstore_data.get("metadatas", []))
            ]
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = TOP_K // 2
            retriever.search_kwargs["k"] = TOP_K // 2
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever], 
                weights=[0.4, 0.6]
            )
            retriever_to_use = ensemble_retriever
        else:
            retriever_to_use = retriever
    except Exception as e:
        log.warning(f"Failed to build BM25 hybrid search, falling back to dense only: {e}")
        retriever_to_use = retriever

    chain = (
        {"context": retriever_to_use | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return chain, retriever_to_use

def clean_answer(text):
    for token in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
        if token in text: text = text.split(token)[0]
    
    import re
    # Force programmatic elimination of markdown artifacts the LM refuses to drop.
    text = re.sub(r'[*#]', '', text)
    return text.strip()

def check_conversational_intercept(text):
    """Intercepts common prompts that small models struggle with or that exceed context boundaries."""
    import re
    text_lower = text.lower().strip()
    
    # 1. Multi-Day Split Intercept
    multi_day_pattern = r'\b([2-7])[- ]?day\b|\b([2-7])\s+days?\b'
    if re.search(multi_day_pattern, text_lower) and "split" in text_lower:
        if not re.search(r'\b(chest|back|legs|arms|shoulders|core|upper|lower|full body|day 1)\b', text_lower):
            return "What muscle group do you want to train on Day 1?"
            
    # 2. Pleasantries Intercept (skips heavy inference)
    if text_lower in ["hello", "hi", "yo", "sup", "hey"]:
        return "Hey there! How can I help you optimize your gains today?"
        
    # 3. Denial Intercept (prevents small-model continuation hallucinations after interruption)
    if text_lower in ["thank you", "thanks", "thanks!", "thank you!", "thx","no", "nope", "no thank you", "no thanks", "nothing"]:
        return "Alright! Let me know when you're ready to continue your workout plan or if you have another question."
        
    return None

def extract_user_stats(llm, text):
    prompt = (
        "<|im_start|>system\n"
        "Extract ONLY explicitly stated fitness stats from the user's message as JSON.\n"
        "Valid keys: weight, weight_unit, height, name, goal, split, body_fat, bench_pr, squat_pr, deadlift_pr.\n"
        "If a stat is NOT mentioned, do NOT include it. Return ONLY a JSON object.\n"
        "Example: 'I weigh 145 lb' -> {\"weight\": 145, \"weight_unit\": \"lb\"}\n"
        "Example: 'my squat is 225' -> {\"squat_pr\": 225}\n"
        "Example: 'hello' -> {}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    try:
        res = llm.invoke(prompt)
        import re
        match = re.search(r"\{.*\}", res, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return {}

def process_extractions(db_manager, stats, text="", user_id=DEFAULT_USER):
    if not stats: return False
    import re as _re
    
    # Fallback: check if the user said "lb" or "pounds" anywhere in the text
    unit_is_lb = bool(_re.search(r'\b(lb|lbs|pound|pounds)\b', str(text).lower()))
    
    w = stats.get("weight")
    bench = stats.get("bench_pr")
    squat = stats.get("squat_pr")
    deadlift = stats.get("deadlift_pr")
    
    # Fallback: if user said "bench" but LLM extracted it as body weight
    if w and bench is None and "bench" in str(text).lower():
        bench = w
        w = None

    if w:
        w = float(w)
        if unit_is_lb or stats.get("weight_unit", "").lower() in ["lb", "lbs", "pound", "pounds"]:
            w = lbs_to_kg(w)
            
    if bench:
        bench = float(bench)
        if unit_is_lb or stats.get("weight_unit", "").lower() in ["lb", "lbs", "pound", "pounds"]:
            bench = lbs_to_kg(bench)
            
    if squat:
        squat = float(squat)
        if unit_is_lb or stats.get("weight_unit", "").lower() in ["lb", "lbs", "pound", "pounds"]:
            squat = lbs_to_kg(squat)
            
    if deadlift:
        deadlift = float(deadlift)
        if unit_is_lb or stats.get("weight_unit", "").lower() in ["lb", "lbs", "pound", "pounds"]:
            deadlift = lbs_to_kg(deadlift)

    db_manager.upsert_profile(user_id, name=stats.get("name"), height_cm=parse_us_height(stats.get("height")) or stats.get("height"), goal=stats.get("goal"))
    db_manager.add_log(
        user_id, 
        weight=w, 
        body_fat=stats.get("body_fat"),
        bench=bench,
        squat=squat,
        deadlift=deadlift,
        split=stats.get("split")
    )
    return True


# ==============================================================
# CLI / Vector Store
# ==============================================================

def build_vectorstore(embeddings):
    if not (CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())):
        raise FileNotFoundError(f"Knowledge base not found at {CHROMA_DIR}. Run 'python vectorize.py'!")
    log.info("Loading ChromaDB...")
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    vectorstore = build_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    db_manager = UserProfileManager(DB_PATH)
    llm = build_llm()
    chain, retriever_chain = build_rag_chain(llm, retriever, db_manager)
    return {"chain": chain, "retriever": retriever_chain, "db_manager": db_manager, "llm": llm}

def interactive_cli(chain, retriever, db_manager, llm):
    print("\n--- Fitness AI Coach Ready ---")
    while True:
        try: q = input("You: ").strip()
        except: break
        if q.lower() in ("exit", "quit"): break
        stats = extract_user_stats(llm, q)
        if stats: process_extractions(db_manager, stats)
        ans = clean_answer(chain.invoke(q))
        print(f"\nCoach: {ans}\n")

def main():
    try:
        sys_comp = initialize_system()
        interactive_cli(sys_comp["chain"], sys_comp["retriever"], sys_comp["db_manager"], sys_comp["llm"])
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
