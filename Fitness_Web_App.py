import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from datetime import datetime
import time
import os

# Import our core logic from Fitness_App
from Fitness_App import initialize_system, DEFAULT_USER, clean_answer, extract_user_stats, process_extractions

# --- Page Config ---
st.set_page_config(
    page_title="Synapse Strength AI", 
    page_icon="💪", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App State & Styling ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Chat"

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    .metric-container { background: #1f2937; padding: 1.5rem; border-radius: 12px; border: 1px solid #374151; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# --- System Initialization ---
@st.cache_resource
def load_fitness_engine():
    """Initializes the LLM and RAG chain once and caches them."""
    try:
        return initialize_system()
    except FileNotFoundError as e:
        st.error("🧠 Fitness Brain Not Found!")
        st.info("Your knowledge base isn't built yet. Please run this command in your terminal first:")
        st.code("python vectorize.py")
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize GPU brain: {e}")
        st.stop()

with st.spinner("🧠 Connecting to Local GPU Brain (Qwen 2.5)..."):
    sys = load_fitness_engine()

# --- Sidebar Logic ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80) 
    st.title("Fitness Identity")
    
    # 1. Bootstrap: Ensure default_user exists in DB
    if not sys["db_manager"].get_latest_stats(DEFAULT_USER)[0]:
        sys["db_manager"].setup_profile(
            user_id=DEFAULT_USER,
            name="Fitness Enthusiast",
            gender="Not Specified",
            birthdate="1995-01-01",
            height_cm=175.0,
            goal="Maintenance"
        )
    
    # 2. Fetch latest profile and log data
    profile, latest_log = sys["db_manager"].get_latest_stats(DEFAULT_USER)
    
    if profile:
        st.success(f"Hello, {profile['name']}!")
        
        # Display Core Metrics
        st.markdown("### Latest Stats")
        if latest_log:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weight", f"{latest_log['weight_kg']} kg")
            with col2:
                st.metric("Body Fat", f"{latest_log['body_fat_pct']}%")
            
            st.divider()
            st.markdown(f"**Goal:** {profile['goal'].capitalize()}")
            st.markdown(f"**Split:** {latest_log['workout_split'] or 'Not set'}")
            
            # PR Badges
            st.markdown("### PR History")
            if latest_log['pr_bench']: 
                st.write(f"🏋️ Bench: {latest_log['pr_bench']}kg x {latest_log['bench_reps'] or 1}")
        else:
            st.info("No log entries yet. Start chatting to log your weight!")
    else:
        st.warning("No profile found. Use the chat to 'setup high-level profile'.")

    st.divider()
    # View Toggle
    mode = st.radio("Navigation", ["Chat with Coach", "Progress Dashboard"])
    st.session_state.view_mode = mode

# --- Progress Dashboard View ---
if st.session_state.view_mode == "Progress Dashboard":
    st.header("📈 Progress Analytics")
    
    # Fetch all logs for charts
    with sqlite3.connect(sys["db_manager"].db_path) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, weight_kg, body_fat_pct, pr_bench FROM logs WHERE user_id = ? ORDER BY timestamp ASC",
            conn, params=(DEFAULT_USER,)
        )
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. Weight Trend Chart
        st.subheader("Weight Trend")
        fig_weight = px.line(df, x='timestamp', y='weight_kg', title='Weight Over Time (kg)', 
                             markers=True, template="plotly_dark")
        fig_weight.update_traces(line_color='#00ff9d')
        st.plotly_chart(fig_weight, use_container_width=True)
        
        # 2. Strength Journey Chart
        if df['pr_bench'].notnull().any():
            st.subheader("Strength Journey (Bench Press)")
            fig_bench = px.area(df.dropna(subset=['pr_bench']), x='timestamp', y='pr_bench', 
                                title='Bench Press PR Over Time', template="plotly_dark")
            fig_bench.update_traces(line_color='#3b82f6')
            st.plotly_chart(fig_bench, use_container_width=True)
    else:
        st.info("Start logging data to see your progress charts!")

# --- Chat Interface View ---
else:
    st.header("🧬 Synapse Strength AI")
    st.caption("Using Qwen-2.5 3B local model & RAG. Ask for workouts based on our Strength, Cardio, and Flexibility datasets.")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            response_box = st.empty()
            
            with st.spinner("Coach is typing..."):
                # 1. Extract Data FIRST (in background)
                new_data = extract_user_stats(sys["llm"], prompt)
                if new_data and any(v is not None for v in new_data.values()):
                    if process_extractions(sys["db_manager"], new_data, DEFAULT_USER):
                        st.toast(f"📊 Progress Auto-logged: {list(new_data.keys())}", icon="✅")
                        # Rebuild context without rerun
                        from Fitness_App import build_rag_chain
                        sys["chain"], _ = build_rag_chain(sys["llm"], sys["retriever"], sys["db_manager"])

                # 2. Answer logic
                from Fitness_App import check_conversational_intercept
                intercept = check_conversational_intercept(prompt)
                
                if intercept:
                    answer = intercept
                else:
                    # Construct generic conversational history for the Streamlit state
                    history_str = ""
                    past_history = st.session_state.messages[-3:-1] # Limit to last 2 messages (1 Q&A pair)
                    if len(past_history) == 2 and past_history[0]["role"] == "user":
                        past_user = past_history[0]["content"]
                        past_coach = past_history[1]["content"]
                        history_str = f"{past_user}\n<|im_end|>\n<|im_start|>assistant\n{past_coach}\n<|im_end|>\n<|im_start|>user\n"
                        
                    enhanced_prompt = f"{history_str}{prompt}"
                    raw_answer = sys["chain"].invoke(enhanced_prompt)
                    answer = clean_answer(raw_answer)
                
                response_box.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # 3. Final cleanup: rerun AT THE END to refresh the sidebar for real
                st.rerun()

st.markdown("---")
st.caption("Privacy: All your data is stored locally in `fitness_app.db` on your RTX 4050 system.")
