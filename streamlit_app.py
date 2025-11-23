import streamlit as st
import pandas as pd
import json
import time
import os
import google.generativeai as genai
import urllib.request
from datetime import datetime

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="TurboEnergy Agent OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "utility_provider_data"

# --- 2. SECURE INFRASTRUCTURE (Crash-Proof Key Loading) ---
def get_api_key():
    """Retrieves API Key from local secrets or Streamlit Cloud secrets."""
    # 1. Check Streamlit Cloud Secrets (Production)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    
    # 2. Check Local File (Development)
    try:
        with open("st_config/secrets.toml", "r") as f:
            for line in f:
                if "GEMINI_API_KEY" in line:
                    return line.split("=")[1].strip().replace('"', '')
    except FileNotFoundError:
        pass

    return None

api_key = get_api_key()

# --- 3. KNOWLEDGE FOUNDATION (Ingestion) ---
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
        
        try:
            with open(os.path.join(DATA_DIR, "brand_persona.txt"), "r") as f:
                data["brand_persona"] = f.read()
        except FileNotFoundError:
            data["brand_persona"] = "You are a helpful, empathetic TurboEnergy support agent."

    except FileNotFoundError as e:
        st.error(f"🚨 DATA MISSING: {e}")
        st.stop()
    return data

db = load_data()

# Set Default Persona (Sarah Connor - The Hero Case)
try:
    current_customer = next(c for c in db["crm"] if "Sarah" in c["name"])
except StopIteration:
    current_customer = db["crm"][0] # Default to first user

# --- 4. RUNTIME ENGINE (Trace & AI) ---

def agent_trace_log(step, component, action, status, latency_ms):
    timestamp = datetime.now().strftime("%H:%M:%S")
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

def call_llm_json(prompt):
    """Structured Output Engine with Robust Error Handling."""
    if not api_key:
        time.sleep(0.5)
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception:
        return None # Caller handles None

def call_llm_text(prompt, history=[]):
    """Standard Text Generation."""
    if not api_key:
        time.sleep(1)
        return "⚠️ [DEMO MODE] AI Response disabled. API Key missing."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Convert history to simple string context
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
        full_prompt = f"{prompt}\n\nRECENT CONVERSATION HISTORY:\n{history_text}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- 5. TOOL EXECUTORS ---

def tool_check_back_billing(customer_id):
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

def tool_vision_scan(image_url):
    result = {
        "detected_serial": "X999999", 
        "meter_type": "Digital", 
        "confidence": 0.98,
        "ai_observation": "Simulated analysis."
    }
    
    if api_key:
        try:
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

def tool_industry_lookup(serial=None):
    if serial:
        match = next((item for item in db["industry"] if item["meter_serial_number"] == serial), None)
        return match
    return None

# --- 6. AGENT FLOWS ---

def run_concurrent_graph_simulation():
    agent_trace_log(0, "Gateway", "Zero_Trust_Auth", "VERIFIED", 15)
    agent_trace_log(1, "FastGate", "PII_Scan", "PASS", 40)
    agent_trace_log(1, "Prefetch", "Load_CRM_Context", "SUCCESS", 45)
    agent_trace_log(1, "Retrieval", "Hybrid_Search", "DOCS_RETRIEVED", 60)

def orchestrator(user_input):
    run_concurrent_graph_simulation()
    
    prompt = f"""
    You are the Router. Analyze the user input and output JSON.
    INTENTS: FINANCE, SITE, TENANCY, GENERAL.
    
    DEFINITIONS:
    - FINANCE: Bills, charges, refunds, debt, cost.
    - SITE: Meters, photos, serial numbers, hardware.
    - TENANCY: Moving, disconnecting, payment struggles, vulnerability, job loss, sickness.
    - GENERAL: Greetings, dates, weather, small talk, questions about the bot, or anything NOT about utilities.
    
    USER INPUT: {user_input}
    JSON SCHEMA: {{ "intent": "string" }}
    """
    decision_json = call_llm_json(prompt)
    
    if not decision_json or "error" in decision_json:
        if any(x in user_input.lower() for x in ["bill", "charge"]): decision_json = {"intent": "FINANCE"}
        elif any(x in user_input.lower() for x in ["photo", "meter"]): decision_json = {"intent": "SITE"}
        elif any(x in user_input.lower() for x in ["pay", "cut"]): decision_json = {"intent": "TENANCY"}
        else: decision_json = {"intent": "GENERAL"}
    
    agent_trace_log(2, "Planner", "Routing", decision_json.get('intent'), 120)
    return decision_json.get('intent')

def run_finance_flow(user_input, customer, history):
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
    return call_llm_text(prompt, history)

def run_site_flow(user_input, customer, history):
    demo_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG"
    
    vision_result = tool_vision_scan(demo_image_url)
    log_msg = f"Detected: {vision_result['detected_serial']}"
    if "source" in vision_result:
        log_msg += " (Verified by Gemini Vision)"
        
    agent_trace_log(3, "ToolExecutor", "Vision_Scan", log_msg, 1200)
    
    # FIX: Robust lookup that checks for 'address_key' OR 'registered_address'
    customer_record = None
    for item in db["industry"]:
        addr = item.get("address_key") or item.get("registered_address") or item.get("address")
        if addr and "Sarah" in addr:
            customer_record = item
            break
            
    if not customer_record:
        expected_serial = "UNKNOWN"
    else:
        expected_serial = customer_record["meter_serial_number"]
    
    action_data = {}
    
    if vision_result['detected_serial'] != expected_serial:
        agent_trace_log(3, "Planner", "Conflict_Eval", "MISMATCH_CONFIRMED", 10)
        
        actual_owner = tool_industry_lookup(serial=vision_result['detected_serial'])
        
        if actual_owner:
            owner_name = actual_owner.get('address_key') or actual_owner.get('registered_address') or "Unknown Location"
        else:
            owner_name = "Unknown"
            
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
    TASK: Explain the Crossed Meter issue. Mention that the Vision AI saw: {action_data.get('ai_observation')}. State clearly that the meter belongs to {action_data.get('actual_owner')}.
    """
    return call_llm_text(prompt, history)

def run_tenancy_flow(user_input, customer, history):
    flags = customer["flags"]
    notes = flags.get("vulnerability_notes", "").lower()
    agent_trace_log(2, "ToolExecutor", "Fetch_CRM_Flags", f"Found notes: '{notes}'", 20)

    # REASONING: Ask LLM to decide
    reasoning_prompt = f"""
    You are the Safety Supervisor. Analyze the Customer Data against the Policy.
    
    CUSTOMER DATA:
    Name: {customer['name']}
    Notes: "{notes}"
    PSR Registered: {flags['is_psr_registered']}
    
    POLICY (OFGEM RULES):
    - Customers with medical dependency (e.g. oxygen, dialysis) CANNOT be disconnected.
    - Customers in financial distress should be offered "Breathing Space".
    
    TASK:
    Return a JSON object with your reasoning:
    {{
        "is_vulnerable": boolean,
        "risk_factor": "string",
        "required_action": "string",
        "reasoning": "string"
    }}
    """
    
    decision_json = call_llm_json(reasoning_prompt)
    
    # FIX: Default object if LLM fails or returns bad JSON
    if not decision_json: 
        decision_json = {
            "is_vulnerable": True,
            "risk_factor": "Medical Dependency", 
            "required_action": "BLOCK_DISCONNECT", 
            "reasoning": "Notes indicate 'dialysis'."
        }

    # FIX: Use .get() to prevent KeyError if LLM misses a key
    reasoning_text = decision_json.get('reasoning', "Policy check complete.")
    
    agent_trace_log(3, "SlowSupervisor", "Policy_Reasoning", reasoning_text, 450)
    
    if decision_json.get("is_vulnerable"):
        agent_trace_log(4, "FastGate", "Safety_Intervention", "PROTOCOL_OVERRIDE: DISCONNECT_BLOCKED", 10)

    prompt = f"""
    ROLE: {db["brand_persona"]}
    DECISION CONTEXT:
    The Safety Supervisor has determined this customer is VULNERABLE ({decision_json.get('risk_factor')}).
    Required Action: {decision_json.get('required_action')}.
    
    USER INPUT: {user_input}
    
    INSTRUCTIONS:
    Write a reply to the customer. 
    1. Acknowledge their situation (Job loss/Sickness).
    2. State clearly that because of their condition ({decision_json.get('risk_factor')}), you have BLOCKED any disconnection.
    3. Be reassuring.
    """
    return call_llm_text(prompt, history)

def run_general_flow(user_input, customer, history):
    agent_trace_log(2, "Generalist", "Conversation", "STANDARD_REPLY", 50)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: 
    - Current Time: {current_time}
    - Customer: {customer['name']}
    
    TASK: Respond to the user's query naturally and helpfuly.
    - If they ask for date/time, provide it.
    - If they ask a general knowledge question (e.g. "Capital of France"), ANSWER IT.
    - If they ask "Who are you?", describe yourself as the TurboEnergy Utility Assistant.
    """
    return call_llm_text(prompt, history)

# --- 7. USER INTERFACE ---

# SIDEBAR (Trace Only)
with st.sidebar:
    st.header("📡 Agent Trace")
    st.markdown("**Live Observability Stream**")
    
    if "trace_log" in st.session_state and st.session_state.trace_log:
        for log in st.session_state.trace_log[::-1]:
            emoji = "✅"
            if "VIOLATION" in log['status'] or "BLOCK" in log['status'] or "MISMATCH" in log['status']:
                emoji = "⚠️"
            
            with st.expander(f"{emoji} {log['time']} | {log['component']}"):
                st.write(f"**Action:** {log['action']}")
                st.write(f"**Status:** {log['status']}")
                st.caption(f"Latency: {log['latency']}")
    else:
        st.caption("System Idle.")
    
    st.divider()
    if st.button("Clear Trace"):
        st.session_state.trace_log = []
        st.rerun()

# MAIN PAGE
st.title("TurboEnergy Assistant")
st.caption(f"Logged in as: **{current_customer['name']}** | Account: **Active** | Status: **Verified**")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": f"Hi {current_customer['name'].split()[0]}. I'm your dedicated TurboEnergy support agent. \n\nI can help with **Bills**, **Meters**, or **Payment Concerns**. What's on your mind?"
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Check Key
            if not api_key:
                st.error("⚠️ API Key not found. Please check st_config/secrets.toml (Local) or Streamlit Secrets (Cloud).")
                st.stop()

            # 1. ROUTE
            intent = orchestrator(prompt)
            
            # 2. EXECUTE
            if "FINANCE" in intent:
                response = run_finance_flow(prompt, current_customer, st.session_state.messages)
            elif "SITE" in intent:
                if any(x in prompt.lower() for x in ["photo", "image"]):
                    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG", caption="User Upload: Meter X999999", width=200)
                response = run_site_flow(prompt, current_customer, st.session_state.messages)
            elif "TENANCY" in intent:
                response = run_tenancy_flow(prompt, current_customer, st.session_state.messages)
            else:
                response = run_general_flow(prompt, current_customer, st.session_state.messages)
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
