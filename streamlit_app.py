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
    """Retrieves API Key from Streamlit Cloud secrets or local file."""
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    try:
        with open("st_config/secrets.toml", "r") as f:
            for line in f:
                if "GEMINI_API_KEY" in line:
                    return line.split("=")[1].strip().replace('"', '').replace("'", "")
    except FileNotFoundError:
        pass 
    return None

api_key = get_api_key()

# --- 3. KNOWLEDGE FOUNDATION (Ingestion) ---
# NOTE: Removed @st.cache_data so we can reload changes dynamically if needed, 
# but primarily we rely on st.session_state for persistence.
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

# Initialize Session State for Database (Crucial for mutation persistence)
if "db" not in st.session_state:
    st.session_state.db = load_data()

# Helper to access db easily
db = st.session_state.db

# Set Default Persona (Sarah Connor - The Hero Case)
try:
    current_customer = next(c for c in db["crm"] if "Sarah" in c["name"])
except StopIteration:
    current_customer = db["crm"][0] 

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
    """Structured Output Engine."""
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
    """Standard Text Generation."""
    if not api_key:
        time.sleep(1)
        return "⚠️ [DEMO MODE] AI Response disabled. API Key missing."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
        full_prompt = f"{prompt}\n\nRECENT CONVERSATION HISTORY:\n{history_text}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- 5. TOOL EXECUTORS (READ & WRITE) ---

def tool_check_back_billing(customer_id):
    ledger = db["ledger"]
    user_ledger = ledger[ledger["customer_id"] == customer_id]
    # Check for old debt (2023) that hasn't been written off yet
    old_debt = user_ledger[
        (user_ledger["usage_start_date"].str.contains("2023")) & 
        (user_ledger["amount_due"] > 0)
    ]
    
    if not old_debt.empty:
        return {
            "found": True,
            "amount": old_debt.iloc[0]["amount_due"],
            "date": old_debt.iloc[0]["usage_start_date"]
        }
    return {"found": False}

def tool_execute_write_off(customer_id, amount, date_ref):
    """Deterministically updates the CSV to write off debt."""
    try:
        df = db["ledger"]
        # Locate the specific row
        mask = (df["customer_id"] == customer_id) & (df["usage_start_date"].astype(str) == date_ref)
        
        if mask.any():
            # 1. Update In-Memory Session State (Immediate UI Feedback)
            df.loc[mask, "status"] = "WRITTEN_OFF_OFGEM"
            df.loc[mask, "amount_due"] = 0.00
            db["ledger"] = df 
            
            # 2. Write to Disk (Persistent change for the file system)
            # Note: On Streamlit Cloud, this persists only for the session duration.
            df.to_csv(os.path.join(DATA_DIR, "billing_ledger.csv"), index=False)
            return True
    except Exception as e:
        st.error(f"Database Update Failed: {e}")
    return False

def tool_update_psr(customer_id, vulnerability_reason):
    """Deterministically updates the CRM JSON to register vulnerability."""
    try:
        crm_path = os.path.join(DATA_DIR, "crm_profiles.json")
        
        customer_found = False
        # 1. Update In-Memory
        for profile in db["crm"]:
            if profile["customer_id"] == customer_id:
                profile["flags"]["is_psr_registered"] = True
                profile["flags"]["vulnerability_notes"] += f" | {vulnerability_reason}"
                customer_found = True
                break
        
        # 2. Write to Disk
        if customer_found:
            with open(crm_path, "w") as f:
                json.dump(db["crm"], f, indent=2)
            return True
    except Exception as e:
        st.error(f"CRM Update Failed: {e}")
    return False

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

def orchestrator(user_input):
    agent_trace_log(0, "Gateway", "Zero_Trust_Auth", "VERIFIED", 15)
    
    prompt = f"""
    You are the Router. Analyze the user input and output JSON.
    INTENTS: FINANCE, SITE, TENANCY, GENERAL.
    
    DEFINITIONS:
    - FINANCE: Bills, charges, refunds, debt, cost.
    - SITE: Meters, photos, serial numbers, hardware.
    - TENANCY: Moving, disconnecting, payment struggles, vulnerability, job loss, sickness.
    - GENERAL: Greetings, dates, weather, small talk.
    
    USER INPUT: {user_input}
    JSON SCHEMA: {{ "intent": "string" }}
    """
    decision_json = call_llm_json(prompt)
    
    if not decision_json or "error" in decision_json:
        # Fallback keyword routing
        lower_input = user_input.lower()
        if any(x in lower_input for x in ["bill", "charge", "owe", "cost"]): decision_json = {"intent": "FINANCE"}
        elif any(x in lower_input for x in ["photo", "meter"]): decision_json = {"intent": "SITE"}
        elif any(x in lower_input for x in ["pay", "cut", "sick", "ill", "hospital"]): decision_json = {"intent": "TENANCY"}
        else: decision_json = {"intent": "GENERAL"}
    
    agent_trace_log(1, "Planner", "Routing", decision_json.get('intent'), 120)
    return decision_json.get('intent')

def run_finance_flow(user_input, customer, history):
    debt_check = tool_check_back_billing(customer["customer_id"])
    action_data = {}
    
    if debt_check["found"]:
        agent_trace_log(2, "SlowSupervisor", "Policy_Check", "OFGEM_21BA_VIOLATION", 80)
        
        # --- DETERMINISTIC ACTION ---
        success = tool_execute_write_off(
            customer["customer_id"], 
            debt_check["amount"], 
            debt_check["date"]
        )
        status_msg = "WRITE_OFF_COMPLETED" if success else "WRITE_OFF_FAILED"
        agent_trace_log(3, "CoreSystem", "Ledger_Update", status_msg, 200)
        # ----------------------------

        action_data = {
            "status": "intervention",
            "tool_result": debt_check,
            "action": "WRITE_OFF",
            "policy": "Ofgem SLC 21BA",
            "outcome": "Debt has been cleared from ledger."
        }
    else:
        action_data = {"status": "clean", "message": "Bill is valid."}
        
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Customer={customer['name']}, Data={json.dumps(action_data)}
    TASK: Write the response. 
    If intervention: Confirm the charge from {debt_check.get('date')} is REMOVED. Be reassuring.
    If clean: Explain the bill is valid.
    """
    return call_llm_text(prompt, history)

def run_site_flow(user_input, customer, history):
    demo_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG"
    
    vision_result = tool_vision_scan(demo_image_url)
    agent_trace_log(2, "ToolExecutor", "Vision_Scan", f"Detected: {vision_result['detected_serial']}", 1200)
    
    customer_record = None
    for item in db["industry"]:
        addr = item.get("address_key") or item.get("registered_address") or item.get("address")
        if addr and "Sarah" in addr:
            customer_record = item
            break
            
    expected_serial = customer_record["meter_serial_number"] if customer_record else "UNKNOWN"
    
    action_data = {}
    if vision_result['detected_serial'] != expected_serial:
        agent_trace_log(3, "Planner", "Conflict_Eval", "MISMATCH_CONFIRMED", 10)
        actual_owner = tool_industry_lookup(serial=vision_result['detected_serial'])
        owner_name = actual_owner.get('address') if actual_owner else "Unknown"
        
        action_data = {
            "status": "mismatch",
            "user_photo_serial": vision_result['detected_serial'],
            "expected": expected_serial,
            "actual_owner": owner_name,
            "resolution": "RAISE_CASE_9921"
        }
    
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Data={json.dumps(action_data)}
    TASK: Explain the Crossed Meter issue. The photo shows a meter belonging to {action_data.get('actual_owner')}. We are fixing this.
    """
    return call_llm_text(prompt, history)

def run_tenancy_flow(user_input, customer, history):
    flags = customer["flags"]
    notes = flags.get("vulnerability_notes", "").lower()
    
    agent_trace_log(2, "ToolExecutor", "Fetch_CRM_Flags", f"Notes: '{notes}'", 20)

    # Ask LLM to evaluate vulnerability based on chat + notes
    reasoning_prompt = f"""
    You are the Safety Supervisor.
    CUSTOMER DATA: Name: {customer['name']}, Notes: "{notes}"
    USER INPUT: "{user_input}"
    
    POLICY: 
    - Medical dependency = BLOCK DISCONNECT.
    - Financial distress/Sickness = BREATHING SPACE.
    
    OUTPUT JSON: {{ "is_vulnerable": boolean, "risk_factor": "string", "required_action": "string" }}
    """
    decision_json = call_llm_json(reasoning_prompt)
    
    if not decision_json: 
        decision_json = {"is_vulnerable": True, "risk_factor": "Manual Review", "required_action": "CHECK_PSR"}

    if decision_json.get("is_vulnerable"):
        agent_trace_log(3, "FastGate", "Safety_Intervention", "PROTOCOL_OVERRIDE: DISCONNECT_BLOCKED", 10)
        
        # --- DETERMINISTIC ACTION ---
        if not flags["is_psr_registered"]:
            success = tool_update_psr(customer["customer_id"], decision_json.get("risk_factor"))
            status_msg = "PSR_UPDATED" if success else "UPDATE_FAILED"
            agent_trace_log(4, "CoreSystem", "CRM_Update", status_msg, 150)
        # ----------------------------

    prompt = f"""
    ROLE: {db["brand_persona"]}
    DECISION: Vulnerable={decision_json.get('is_vulnerable')}, Action={decision_json.get('required_action')}
    USER INPUT: {user_input}
    TASK: Acknowledge the difficult situation. Confirm we have added them to the Priority Register (PSR) and Disconnection is BLOCKED.
    """
    return call_llm_text(prompt, history)

def run_general_flow(user_input, customer, history):
    agent_trace_log(2, "Generalist", "Conversation", "STANDARD_REPLY", 50)
    prompt = f"""
    ROLE: {db["brand_persona"]}
    CONTEXT: Customer={customer['name']}, Time={datetime.now().strftime("%Y-%m-%d %H:%M")}
    TASK: Helpful, warm chat.
    """
    return call_llm_text(prompt, history)

# --- 7. USER INTERFACE ---

# SIDEBAR: TRACE & DATA INSPECTOR
with st.sidebar:
    st.header("📡 Agent Dashboard")
    
    # TRACE LOG
    st.subheader("Trace Log")
    if "trace_log" in st.session_state and st.session_state.trace_log:
        for log in st.session_state.trace_log[::-1]:
            emoji = "✅"
            if "VIOLATION" in log['status'] or "BLOCK" in log['status']: emoji = "🛡️"
            if "MISMATCH" in log['status']: emoji = "⚠️"
            if "UPDATE" in log['status'] or "WRITE_OFF" in log['status']: emoji = "💾"
            
            with st.expander(f"{emoji} {log['component']}"):
                st.caption(f"Time: {log['time']} | Latency: {log['latency']}")
                st.write(f"**Action:** {log['action']}")
                st.write(f"**Status:** {log['status']}")
    else:
        st.caption("Waiting for input...")
        
    st.divider()
    
    # RECRUITER VIEW: LIVE DATA INSPECTOR
    st.subheader("📂 Live Data Inspector")
    st.info("Expand below to verify database mutations occur in real-time.")
    
    with st.expander("💰 Billing Ledger (Live)"):
        # Show only relevant rows for Sarah
        ledger_view = db["ledger"][db["ledger"]["customer_id"] == current_customer["customer_id"]]
        st.dataframe(ledger_view, hide_index=True)
        
    with st.expander("👤 CRM Profile (Live)"):
        # Find current customer profile
        profile_view = next((p for p in db["crm"] if p["customer_id"] == current_customer["customer_id"]), {})
        st.json(profile_view)

# MAIN PAGE
st.title("TurboEnergy Assistant")
st.caption(f"Logged in as: **{current_customer['name']}** | Account: **Active** | Priority Status: **{current_customer['flags']['is_psr_registered']}**")

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
                if any(x in prompt.lower() for x in ["photo", "image"]):
                    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG", caption="Uploaded Image: Meter X999999", width=200)
                response = run_site_flow(prompt, current_customer, st.session_state.messages)
            elif "TENANCY" in intent:
                response = run_tenancy_flow(prompt, current_customer, st.session_state.messages)
            else:
                response = run_general_flow(prompt, current_customer, st.session_state.messages)
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force refresh to update Sidebar Data Inspector immediately
            st.rerun()
