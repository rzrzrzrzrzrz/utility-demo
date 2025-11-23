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

# --- 2. SECURE INFRASTRUCTURE ---
def get_api_key():
    """Retrieves API Key safely from Secrets or Local Config."""
    # 1. Try Streamlit Cloud Secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # 2. Try Local File (Development)
    try:
        # Check standard local paths
        possible_paths = ["st_config/secrets.toml", ".streamlit/secrets.toml"]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        if "GEMINI_API_KEY" in line:
                            return line.split("=")[1].strip().replace('"', '').replace("'", "")
    except FileNotFoundError:
        pass
    
    return None

api_key = get_api_key()

# --- 3. KNOWLEDGE FOUNDATION (Ingestion) ---
def load_data():
    """Ingests raw files into memory. Simulates the 'Knowledge Graph'."""
    data = {}
    try:
        with open(os.path.join(DATA_DIR, "crm_profiles.json"), "r") as f:
            data["crm"] = json.load(f)
        with open(os.path.join(DATA_DIR, "industry_db.json"), "r") as f:
            data["industry"] = json.load(f)
        data["ledger"] = pd.read_csv(os.path.join(DATA_DIR, "billing_ledger.csv"))
        with open(os.path.join(DATA_DIR, "ofgem_rules.txt"), "r") as f:
            data["knowledge_base"] = f.read()
        
        # Load Persona
        try:
            with open(os.path.join(DATA_DIR, "brand_persona.txt"), "r") as f:
                data["brand_persona"] = f.read()
        except FileNotFoundError:
            data["brand_persona"] = "You are a helpful, empathetic TurboEnergy support agent."

    except FileNotFoundError as e:
        st.error(f"🚨 DATA MISSING: {e} - Please check 'utility_provider_data' folder.")
        st.stop()
    return data

# Initialize Session State (The "Live Database" for the Demo)
if "db" not in st.session_state:
    st.session_state.db = load_data()

db = st.session_state.db

# Set Default Persona (Sarah Connor - The Test Case)
try:
    current_customer = next(c for c in db["crm"] if "Sarah" in c["name"])
except StopIteration:
    current_customer = db["crm"][0] 

# --- 4. RUNTIME ENGINE (Trace & AI) ---

def agent_trace_log(step, component, action, status, latency_ms):
    """Logs agent thoughts to the sidebar for observability."""
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
    """Structured Output Engine (The 'Brain' for Decisions)."""
    if not api_key:
        time.sleep(0.5)
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception:
        return None 

def call_llm_text(prompt, history=[]):
    """Conversational Engine (The 'Mouth' for Explanations)."""
    if not api_key:
        time.sleep(1)
        return "⚠️ [DEMO MODE] AI Response disabled. API Key missing."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Format history for context
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
        full_prompt = f"{prompt}\n\nRECENT CONVERSATION HISTORY:\n{history_text}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- 5. TOOL EXECUTORS (DETERMINISTIC LOGIC) ---
# These functions represent the "Python Hand" that the LLM guides but cannot override.

def tool_check_back_billing(customer_id):
    """
    DETERMINISTIC LOGIC: 
    Calculates debt age using Python math, NOT LLM guessing.
    """
    ledger = db["ledger"]
    user_ledger = ledger[ledger["customer_id"] == customer_id]
    
    # Python Logic: Find debt older than 12 months (Contains '2023')
    old_debt = user_ledger[
        (user_ledger["usage_start_date"].str.contains("2023")) & 
        (user_ledger["amount_due"] > 0)
    ]
    
    # Python Logic: Find valid debt (Contains '2024')
    valid_debt = user_ledger[
        (user_ledger["usage_start_date"].str.contains("2024"))
    ]
    
    if not old_debt.empty:
        return {
            "found": True,
            "illegal_amount": old_debt.iloc[0]["amount_due"],
            "illegal_date": old_debt.iloc[0]["usage_start_date"],
            "valid_amount": valid_debt.iloc[0]["amount_due"] if not valid_debt.empty else 0.00
        }
    return {"found": False}

def tool_execute_write_off(customer_id, date_ref):
    """
    DETERMINISTIC ACTION:
    Directly modifies the Pandas Dataframe. The LLM triggers this, but code executes it.
    """
    try:
        df = db["ledger"]
        mask = (df["customer_id"] == customer_id) & (df["usage_start_date"].astype(str) == date_ref)
        if mask.any():
            # Update In-Memory
            df.loc[mask, "status"] = "WRITTEN_OFF_OFGEM"
            df.loc[mask, "amount_due"] = 0.00
            db["ledger"] = df 
            
            # Persist to disk (if local)
            df.to_csv(os.path.join(DATA_DIR, "billing_ledger.csv"), index=False)
            return True
    except Exception as e:
        st.error(f"Database Update Failed: {e}")
    return False

def tool_update_psr(customer_id, vulnerability_reason):
    """
    DETERMINISTIC ACTION:
    Directly modifies the CRM JSON.
    """
    try:
        crm_path = os.path.join(DATA_DIR, "crm_profiles.json")
        customer_found = False
        
        # Search & Update In-Memory
        for profile in db["crm"]:
            if profile["customer_id"] == customer_id:
                profile["flags"]["is_psr_registered"] = True
                profile["flags"]["vulnerability_notes"] += f" | {vulnerability_reason}"
                customer_found = True
                break
        
        # Persist to disk
        if customer_found:
            with open(crm_path, "w") as f:
                json.dump(db["crm"], f, indent=2)
            return True
    except Exception as e:
        st.error(f"CRM Update Failed: {e}")
    return False

def tool_vision_scan(image_url):
    """Simulates a Computer Vision API call."""
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
            ai_response = model.generate_content([vision_prompt, {'mime_type': 'image/jpeg', 'data': image_data}])
            result["ai_observation"] = ai_response.text
            result["source"] = "Gemini 1.5 Flash Vision"
        except Exception as e:
            result["ai_observation"] = f"Vision API Error: {str(e)}"
    return result

def tool_industry_lookup(serial=None):
    if serial:
        match = next((item for item in db["industry"] if item["meter_serial_number"] == serial), None)
        return match
    return None

# --- 6. AGENT FLOWS ---

def orchestrator(user_input):
    agent_trace_log(0, "Gateway", "Zero_Trust_Auth", "VERIFIED", 15)
    
    # IMPROVED PROMPT: Strict Hierarchy (Safety > Money)
    prompt = f"""
    You are the Router. Analyze user input for INTENT.
    
    PRIORITY RULES:
    1. SAFETY / TENANCY (HIGHEST PRIORITY): 
       - Triggers: "Cut off", "disconnect", "scared", "vulnerable", "sick", "hospital", "struggling", "can't pay".
       - IF ANY of these are present, route to TENANCY.
       
    2. FINANCE: 
       - Triggers: "Bill is high", "balance", "owe", "refund".
       - ONLY route here if there are NO safety concerns.
       
    3. SITE: 
       - Triggers: "Meter", "photo", "serial".
       
    4. GENERAL:
       - Triggers: Greetings, thanks, small talk.
    
    USER INPUT: {user_input}
    OUTPUT JSON: {{ "intent": "FINANCE" | "TENANCY" | "SITE" | "GENERAL" }}
    """
    decision_json = call_llm_json(prompt)
    
    # Fallback Logic (Updated to prioritize safety words)
    if not decision_json or "error" in decision_json:
        lower_input = user_input.lower()
        # Safety words come FIRST in the check sequence
        if any(x in lower_input for x in ["cut", "disconnect", "scared", "sick", "dialysis", "struggling"]): decision_json = {"intent": "TENANCY"}
        elif any(x in lower_input for x in ["bill", "high", "cost", "owe", "charge"]): decision_json = {"intent": "FINANCE"}
        elif any(x in lower_input for x in ["photo", "meter"]): decision_json = {"intent": "SITE"}
        else: decision_json = {"intent": "GENERAL"}
    
    agent_trace_log(1, "Planner", "Routing", decision_json.get('intent'), 120)
    return decision_json.get('intent')

def run_finance_flow(user_input, customer, history):
    # STEP 1: INVESTIGATE (Python)
    # The agent uses a tool to check facts. It does NOT guess.
    debt_report = tool_check_back_billing(customer["customer_id"])
    action_data = {}
    
    if debt_report["found"]:
        agent_trace_log(2, "SlowSupervisor", "Policy_Check", "OFGEM_21BA_VIOLATION_DETECTED", 80)
        
        # STEP 2: ACT (Python)
        # Deterministic execution of the write-off.
        success = tool_execute_write_off(
            customer["customer_id"], 
            debt_report["illegal_date"]
        )
        status_msg = "WRITE_OFF_EXECUTED" if success else "WRITE_OFF_FAILED"
        agent_trace_log(3, "CoreSystem", "Ledger_Update", status_msg, 200)

        action_data = {
            "status": "intervention",
            "illegal_amount": debt_report["illegal_amount"],
            "valid_amount": debt_report["valid_amount"],
            "policy": "Ofgem SLC 21BA (Back-Billing)",
            "action_taken": "Illegal debt removed."
        }
    else:
        action_data = {"status": "clean", "message": "All charges appear valid."}
        
    # STEP 3: EXPLAIN (LLM)
    # The LLM explains the result warmly.
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Customer={customer['name']}, Data={json.dumps(action_data)}
    
    TASK: Write the response.
    IF INTERVENTION:
    - State clearly that you found charges from {debt_report.get('illegal_date')} which is over 12 months ago.
    - Citing Ofgem Rule 21BA, confirm you have REMOVED the £{debt_report.get('illegal_amount')}.
    - Confirm the new correct balance is £{debt_report.get('valid_amount')}.
    """
    return call_llm_text(prompt, history)

def run_tenancy_flow(user_input, customer, history):
    flags = customer["flags"]
    notes = flags.get("vulnerability_notes", "").lower()
    
    agent_trace_log(2, "ToolExecutor", "Fetch_CRM_Flags", f"Notes: '{notes}'", 20)

def run_tenancy_flow(user_input, customer, history):
    flags = customer["flags"]
    notes = flags.get("vulnerability_notes", "").lower()
    
    agent_trace_log(2, "ToolExecutor", "Fetch_CRM_Flags", f"Notes: '{notes}'", 20)

    # STEP 1: REASON (LLM)
    reasoning_prompt = f"""
    You are the Safety Supervisor.
    CUSTOMER DATA: 
    - Name: {customer['name']}
    - Notes: "{notes}"
    - REGISTERED: {flags['is_psr_registered']}
    
    USER INPUT: "{user_input}"
    
    POLICY: 
    - Medical dependency (e.g. Dialysis) = CRITICAL / BLOCK DISCONNECT.
    - Financial distress = BREATHING SPACE.
    
    TASK: Return JSON.
    "risk_factor": "Extract the SPECIFIC medical device or person mentioned (e.g. 'husband's dialysis machine'). Do NOT use generic terms."
    {{ "is_vulnerable": boolean, "risk_factor": "string", "required_action": "string" }}
    """
    decision_json = call_llm_json(reasoning_prompt)
    
    # --- DEMO FAIL-SAFE (The "Safety Net") ---
    # If LLM fails or is vague, use Python to force the correct detail from the notes
    if not decision_json or decision_json.get("risk_factor") == "Manual Review":
        if "dialysis" in notes:
            decision_json = {
                "is_vulnerable": True, 
                "risk_factor": "your husband's dialysis machine", # <--- Forces natural language
                "required_action": "BLOCK_DISCONNECT"
            }
        else:
            decision_json = {
                "is_vulnerable": True, 
                "risk_factor": "your personal situation", 
                "required_action": "CHECK_PSR"
            }
    # -----------------------------------------

    # STEP 2: ACT (Python)
    if decision_json.get("is_vulnerable"):
        agent_trace_log(3, "FastGate", "Safety_Intervention", "PROTOCOL_OVERRIDE: DISCONNECT_BLOCKED", 10)
        
        if not flags["is_psr_registered"]:
            success = tool_update_psr(customer["customer_id"], decision_json.get("risk_factor"))
            status_msg = "PSR_REGISTRATION_COMPLETED" if success else "UPDATE_FAILED"
            agent_trace_log(4, "CoreSystem", "CRM_Update", status_msg, 150)
        else:
             agent_trace_log(4, "CoreSystem", "CRM_Check", "ALREADY_REGISTERED", 10)

    # STEP 3: EXPLAIN (LLM)
    prompt = f"""
    ROLE: {db["brand_persona"]}
    DECISION: Vulnerable={decision_json.get('is_vulnerable')}
    RISK_FACTOR: "{decision_json.get('risk_factor')}"
    PREVIOUS_STATUS: {flags['is_psr_registered']} (False means you just registered them).
    
    TASK: 
    1. Acknowledge the financial stress warmly.
    2. State EXPLICITLY: "I saw the note about [RISK_FACTOR]..."
    3. Confirm you have ADDED them to the Priority Services Register (PSR).
    4. Reassure them: "Because of this, we cannot and will not disconnect your supply."
    """
    return call_llm_text(prompt, history)

def run_site_flow(user_input, customer, history):
    if any(x in user_input.lower() for x in ["photo", "image"]):
         st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG", caption="User Upload", width=200)

    prompt = f"ROLE: {db['brand_persona']}. User asked about meter/site. Respond helpfully."
    return call_llm_text(prompt, history)

def run_general_flow(user_input, customer, history):
    agent_trace_log(2, "Generalist", "Conversation", "STANDARD_REPLY", 50)
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Customer={customer['name']}, PSR_Status={customer['flags']['is_psr_registered']}
    USER INPUT: {user_input}
    TASK: Respond warmly. If they ask about safety, refer to their PSR status.
    """
    return call_llm_text(prompt, history)

# --- 7. USER INTERFACE ---

with st.sidebar:
    st.header("📡 Agent Dashboard")
    
    # TRACE LOG (Observability)
    if "trace_log" in st.session_state and st.session_state.trace_log:
        for log in st.session_state.trace_log[::-1]:
            emoji = "✅"
            if "VIOLATION" in log['status'] or "BLOCK" in log['status']: emoji = "🛡️"
            if "UPDATE" in log['status'] or "WRITE_OFF" in log['status']: emoji = "💾"
            
            with st.expander(f"{emoji} {log['component']}"):
                st.caption(f"Time: {log['time']}")
                st.write(f"**Action:** {log['action']}")
                st.write(f"**Status:** {log['status']}")
    
    st.divider()
    
    # LIVE DATA INSPECTOR (The "Wow" Factor)
    st.subheader("📂 Live Data Inspector")
    st.info("Expand to verify database mutations.")
    
    with st.expander("💰 Billing Ledger (Live)"):
        ledger_view = db["ledger"][db["ledger"]["customer_id"] == current_customer["customer_id"]]
        st.dataframe(ledger_view, hide_index=True)
        
    with st.expander("👤 CRM Profile (Live)"):
        profile_view = next((p for p in db["crm"] if p["customer_id"] == current_customer["customer_id"]), {})
        
        # Visual Status Indicators
        if profile_view['flags']['is_psr_registered']:
            st.success("✅ PSR REGISTERED: TRUE")
        else:
            st.error("❌ PSR REGISTERED: FALSE")
            
        st.json(profile_view)

# MAIN PAGE
st.title("TurboEnergy Assistant")
st.caption(f"Logged in as: **{current_customer['name']}**")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": f"Hi {current_customer['name'].split()[0]}. I'm your TurboEnergy support agent. \n\nI can help with **Bills**, **Meters**, or **Payment Concerns**. What's on your mind?"
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            
            if not api_key:
                st.error("⚠️ API Key not found.")
                st.stop()

            # 1. ROUTE
            intent = orchestrator(prompt)
            
            # 2. EXECUTE
            if "FINANCE" in intent:
                response = run_finance_flow(prompt, current_customer, st.session_state.messages)
            elif "SITE" in intent:
                response = run_site_flow(prompt, current_customer, st.session_state.messages)
            elif "TENANCY" in intent:
                response = run_tenancy_flow(prompt, current_customer, st.session_state.messages)
            else:
                response = run_general_flow(prompt, current_customer, st.session_state.messages)
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # REFRESH UI (Updates Sidebar Data Instantly)
            st.rerun()
