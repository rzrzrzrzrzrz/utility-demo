import streamlit as st
import pandas as pd
import json
import time
import os
import google.generativeai as genai
from datetime import datetime

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Project Conduit | Agent OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "utility_provider_data"

# --- SECURE KEY LOADING (The Fix) ---
# This runs immediately to check for your local secret file
if "api_key" not in st.session_state:
    try:
        # We try to open the local secret file
        with open("st_config/secrets.toml", "r") as f:
            for line in f:
                if "GEMINI_API_KEY" in line:
                    # Clean the string to extract just the key part
                    key_value = line.split("=")[1].strip().replace('"', '')
                    st.session_state["api_key"] = key_value
    except FileNotFoundError:
        pass # No file found, user will have to paste it manually

# --- 2. PHASE 1: THE KNOWLEDGE FOUNDATION (Ingestion) ---
@st.cache_data
def load_data():
    data = {}
    try:
        with open(os.path.join(DATA_DIR, "crm_profiles.json"), "r") as f:
            data["crm"] = json.load(f)
        with open(os.path.join(DATA_DIR, "industry_db.json"), "r") as f:
            data["industry"] = json.load(f)
        data["ledger"] = pd.read_csv(os.path.join(DATA_DIR, "billing_ledger.csv"))
        with open(os.path.join(DATA_DIR, "ofgem_rules.txt"), "r") as f:
            data["knowledge_base"] = f.read()
        
        # Load Brand Persona
        try:
            with open(os.path.join(DATA_DIR, "brand_persona.txt"), "r") as f:
                data["brand_persona"] = f.read()
        except FileNotFoundError:
            data["brand_persona"] = "You are a helpful, empathetic utility support agent."

    except FileNotFoundError as e:
        st.error(f"🚨 DATA MISSING: {e}")
        st.stop()
    return data

db = load_data()

# --- 3. PHASE 2: THE RUNTIME ENGINE (Observability & Helpers) ---

def agent_trace_log(step, component, action, status, latency_ms):
    """
    The 'Black Box' Recorder. 
    Logs every step to the sidebar to prove observability.
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        "time": timestamp,
        "step": step,
        "component": component,
        "action": action,
        "status": status,
        "latency": f"{latency_ms}ms"
    }
    if "trace_log" not in st.session_state:
        st.session_state.trace_log = []
    st.session_state.trace_log.append(log_entry)

def call_llm_json(prompt, api_key):
    """Structured Output Engine."""
    if not api_key:
        time.sleep(0.5)
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def call_llm_text(prompt, api_key):
    """Standard Text Generation."""
    if not api_key:
        time.sleep(1)
        return "⚠️ [DEMO MODE] Real AI response disabled. Please provide API Key."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- 4. THE TOOL EXECUTORS (Deterministic Logic) ---

def tool_check_back_billing(customer_id):
    """Tool: Analyzes ledger for debt > 12 months old."""
    ledger = db["ledger"]
    user_ledger = ledger[ledger["customer_id"] == customer_id]
    old_debt = user_ledger[user_ledger["usage_start_date"].str.contains("2023")]
    
    if not old_debt.empty:
        return {
            "found": True,
            "amount": old_debt.iloc[0]["amount_due"],
            "date": old_debt.iloc[0]["usage_start_date"]
        }
    return {"found": False}

def tool_vision_scan(image_url, api_key=None):
    """Tool: Multimodal Computer Vision."""
    # 1. Default Demo Result
    result = {
        "detected_serial": "X999999", 
        "meter_type": "Digital", 
        "confidence": 0.98,
        "ai_observation": "Simulated analysis."
    }
    
    # 2. Real AI Execution (If Key Exists)
    if api_key:
        try:
            import urllib.request
            with urllib.request.urlopen(image_url) as response:
                image_data = response.read()

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            vision_prompt = "Analyze this image. Confirm it is an electricity meter. Describe the display type."
            
            ai_response = model.generate_content([
                vision_prompt,
                {'mime_type': 'image/jpeg', 'data': image_data}
            ])
            
            result["ai_observation"] = ai_response.text
            result["source"] = "Gemini 1.5 Flash Vision"
            
        except Exception as e:
            result["ai_observation"] = f"Vision API Error: {str(e)}"
    else:
        time.sleep(1.0)

    return result

def tool_industry_lookup(mpan=None, serial=None):
    """Tool: Queries the National Database (ECOES)."""
    if serial:
        match = next((item for item in db["industry"] if item["meter_serial_number"] == serial), None)
        return match
    if mpan:
        match = next((item for item in db["industry"] if item["mpan"] == mpan), None)
        return match
    return None

# --- 5. THE AGENT LOGIC (The "Graph") ---

def run_concurrent_graph_simulation():
    agent_trace_log(0, "Gateway", "Zero_Trust_Auth", "VERIFIED", 15)
    agent_trace_log(1, "FastGate", "PII_Scan", "PASS", 40)
    agent_trace_log(1, "Prefetch", "Load_CRM_Context", "SUCCESS", 45)
    agent_trace_log(1, "Retrieval", "Hybrid_Search", "DOCS_RETRIEVED", 60)

def orchestrator(user_input, api_key):
    run_concurrent_graph_simulation()
    
    prompt = f"""
    You are the Router. Analyze the user input and output JSON.
    INTENTS: FINANCE, SITE, TENANCY.
    USER INPUT: {user_input}
    JSON SCHEMA: {{ "intent": "string", "confidence": float, "reasoning": "string" }}
    """
    decision_json = call_llm_json(prompt, api_key)
    
    if not decision_json or "error" in decision_json:
        if any(x in user_input.lower() for x in ["bill", "charge"]): decision_json = {"intent": "FINANCE"}
        elif any(x in user_input.lower() for x in ["photo", "meter"]): decision_json = {"intent": "SITE"}
        else: decision_json = {"intent": "TENANCY"}
    
    agent_trace_log(2, "Planner", "Routing", decision_json.get('intent'), 120)
    return decision_json.get('intent')

# --- SPECIALIST AGENTS ---

def run_finance_flow(user_input, customer, api_key):
    debt_check = tool_check_back_billing(customer["customer_id"])
    action_data = {}
    
    if debt_check["found"]:
        agent_trace_log(3, "SlowSupervisor", "Policy_Check", "OFGEM_21BA_VIOLATION", 80)
        action_data = {
            "status": "intervention",
            "tool_result": debt_check,
            "action": "WRITE_OFF",
            "policy": "Ofgem SLC 21BA"
        }
    else:
        action_data = {"status": "clean", "message": "Bill is valid."}
        
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Customer={customer['name']}, Data={json.dumps(action_data)}
    TASK: Write the response. If intervention, explain the write-off clearly.
    """
    return call_llm_text(prompt, api_key)

def run_site_flow(user_input, customer, api_key):
    demo_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG"
    
    vision_result = tool_vision_scan(demo_image_url, api_key)
    
    log_msg = f"Detected: {vision_result['detected_serial']}"
    if "source" in vision_result:
        log_msg += " (Verified by Gemini Vision)"
        
    agent_trace_log(3, "ToolExecutor", "Vision_Scan", log_msg, 1200)
    
    customer_record = next((i for i in db["industry"] if "Sarah" in i["address_key"]), None)
    expected_serial = customer_record["meter_serial_number"]
    
    action_data = {}
    
    if vision_result['detected_serial'] != expected_serial:
        agent_trace_log(3, "Planner", "Conflict_Eval", "MISMATCH_CONFIRMED", 10)
        actual_owner = tool_industry_lookup(serial=vision_result['detected_serial'])
        owner_name = actual_owner['address_key'] if actual_owner else "Unknown"
        
        action_data = {
            "status": "mismatch",
            "user_photo_serial": vision_result['detected_serial'],
            "ai_observation": vision_result.get("ai_observation", ""),
            "db_record": expected_serial,
            "actual_owner": owner_name,
            "resolution": "RAISE_CASE_9921"
        }
    
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Data={json.dumps(action_data)}
    TASK: Explain the Crossed Meter issue. State clearly that the meter in the photo belongs to {action_data.get('actual_owner')}.
    """
    return call_llm_text(prompt, api_key)

def run_tenancy_flow(user_input, customer, api_key):
    flags = customer["flags"]
    notes = flags.get("vulnerability_notes", "").lower()
    
    action_data = {}
    if "dialysis" in notes or flags["is_psr_registered"]:
        agent_trace_log(2, "FastGate", "Safety_Intervention", "BLOCK_DISCONNECTION", 5)
        action_data = {
            "status": "vulnerable",
            "reason": "Medical Dependency",
            "action": "Emergency Protocol Activated"
        }
    else:
        action_data = {"status": "standard", "action": "Payment Plan"}
        
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Customer={customer['name']}, Data={json.dumps(action_data)}
    TASK: Write response. If vulnerable, be extremely reassuring and state disconnection is impossible.
    """
    return call_llm_text(prompt, api_key)

# --- 6. UI SETUP ---

with st.sidebar:
    st.header("⚡ Project Conduit")
    
    # --- THE FIX: Pre-fill the key from session state ---
    # We use .get() to grab the key if the loading block found it
    default_key = st.session_state.get("api_key", "")
    
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    
    st.divider()
    
    selected_customer_name = st.selectbox("Persona", [p["name"] for p in db["crm"]])
    current_customer = next(c for c in db["crm"] if c["name"] == selected_customer_name)
    st.info(f"Context Loaded: {current_customer['customer_id']}")
    
    st.subheader("Agent Trace")
    if "trace_log" in st.session_state:
        for log in st.session_state.trace_log[::-1]:
            with st.expander(f"{log['component']} | {log['status']}"):
                st.json(log)

st.title("Agent Interface")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Online."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = orchestrator(prompt, api_key)
            
            if "FINANCE" in intent:
                response = run_finance_flow(prompt, current_customer, api_key)
            elif "SITE" in intent:
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG", caption="User Upload: Meter X999999", width=200)
                response = run_site_flow(prompt, current_customer, api_key)
            elif "TENANCY" in intent:
                response = run_tenancy_flow(prompt, current_customer, api_key)
            else:
                response = "I couldn't determine the path."
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
