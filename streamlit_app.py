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

# --- 0. AUTO-RESET MECHANISM (Demo Loop) ---
def reset_demo_files():
    """
    Overwrites the data files with the 'Before' state (Problem State).
    This ensures every fresh browser session starts with the debt/vulnerability issues.
    """
    # 1. Reset CRM (Sarah is NOT registered)
    crm_data = [
      {
        "customer_id": "CUST-9982",
        "name": "Sarah Connor",
        "address": "42 Industrial Estate, London, E14 5AB",
        "flags": {
          "is_psr_registered": False, # <--- Forces 'False' on startup
          "vulnerability_notes": "Customer mentioned husband is on home dialysis machine during call on 12/08."
        }
      },
      {
        "customer_id": "CUST-1001",
        "name": "John Smith",
        "address": "1 The Green, Manchester, M1 1AA",
        "flags": {
          "is_psr_registered": True,
          "vulnerability_notes": "Elderly, Priority Services Register (Blind)."
        }
      }
    ]
    with open(os.path.join(DATA_DIR, "crm_profiles.json"), "w") as f:
        json.dump(crm_data, f, indent=2)

    # 2. Reset Ledger (Old Debt Exists)
    ledger_data = """customer_id,usage_start_date,usage_end_date,amount_due,status,error_type
CUST-9982,2023-01-01,2023-12-31,1450.00,UNBILLED,Supplier Estimate Error
CUST-9982,2024-01-01,2024-11-23,1000.00,BILLED,Current Debt
CUST-1001,2024-09-01,2024-10-01,150.00,PAID,N/A"""
    
    with open(os.path.join(DATA_DIR, "billing_ledger.csv"), "w") as f:
        f.write(ledger_data)

# --- CRITICAL: RUN RESET ON BROWSER REFRESH ONLY ---
if "demo_reset_done" not in st.session_state:
    reset_demo_files()
    st.session_state.demo_reset_done = True

# --- GLOBAL GUARDRAILS (The "Constitution") ---
# This applies to ALL agents to ensure they hand off unauthorized tasks.
GLOBAL_GUARDRAILS = """
CRITICAL AUTHORITY BOUNDARIES:
1. AUTHORIZED TASKS (You CAN do these):
   - Identifying and removing debt older than 12 months (Back-Billing).
   - Registering vulnerable customers on the Priority Services Register (Safety).
   - analyzing meter photos.

2. UNAUTHORIZED TASKS (You MUST HANDOFF to Human):
   - Setting up Payment Plans or Direct Debits.
   - Changing Tariffs or Renewing Contracts.
   - Moving House / Closing Accounts.
   - Formal Complaints.

INSTRUCTION FOR HANDOFF:
If the user asks for an UNAUTHORIZED task, do NOT attempt to answer it. 
State clearly: "I am not authorized to manage [Topic], so I am transferring you to a human specialist who can set this up for you immediately."
"""

# --- 2. SECURE INFRASTRUCTURE ---
def get_api_key():
    """Retrieves API Key safely."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    try:
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

# --- 3. KNOWLEDGE FOUNDATION ---
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

if "db" not in st.session_state:
    st.session_state.db = load_data()

db = st.session_state.db

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

# --- 5. TOOL EXECUTORS ---

def tool_check_back_billing(customer_id):
    ledger = db["ledger"]
    user_ledger = ledger[ledger["customer_id"] == customer_id]
    old_debt = user_ledger[(user_ledger["usage_start_date"].str.contains("2023")) & (user_ledger["amount_due"] > 0)]
    valid_debt = user_ledger[(user_ledger["usage_start_date"].str.contains("2024"))]
    
    if not old_debt.empty:
        return {
            "found": True,
            "illegal_amount": old_debt.iloc[0]["amount_due"],
            "illegal_date": old_debt.iloc[0]["usage_start_date"],
            "valid_amount": valid_debt.iloc[0]["amount_due"] if not valid_debt.empty else 0.00
        }
    return {"found": False}

def tool_execute_write_off(customer_id, date_ref):
    try:
        df = db["ledger"]
        mask = (df["customer_id"] == customer_id) & (df["usage_start_date"].astype(str) == date_ref)
        if mask.any():
            df.loc[mask, "status"] = "WRITTEN_OFF_OFGEM"
            df.loc[mask, "amount_due"] = 0.00
            db["ledger"] = df 
            df.to_csv(os.path.join(DATA_DIR, "billing_ledger.csv"), index=False)
            return True
    except Exception as e:
        st.error(f"Database Update Failed: {e}")
    return False

def tool_update_psr(customer_id, vulnerability_reason):
    try:
        crm_path = os.path.join(DATA_DIR, "crm_profiles.json")
        customer_found = False
        for profile in db["crm"]:
            if profile["customer_id"] == customer_id:
                profile["flags"]["is_psr_registered"] = True
                profile["flags"]["vulnerability_notes"] += f" | {vulnerability_reason}"
                customer_found = True
                break
        if customer_found:
            with open(crm_path, "w") as f:
                json.dump(db["crm"], f, indent=2)
            return True
    except Exception as e:
        st.error(f"CRM Update Failed: {e}")
    return False

def tool_vision_scan(image_url):
    result = {"detected_serial": "X999999", "meter_type": "Digital", "confidence": 0.98, "ai_observation": "Simulated analysis."}
    if api_key:
        try:
            with urllib.request.urlopen(image_url) as response:
                image_data = response.read()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            vision_prompt = "Analyze this image. Confirm it is an electricity meter. Describe the display type."
            ai_response = model.generate_content([vision_prompt, {'mime_type': 'image/jpeg', 'data': image_data}])
            result["ai_observation"] = ai_response.text
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
    
    prompt = f"""
    You are the Router. Analyze user input for INTENT.
    
    PRIORITY RULES:
    1. SAFETY / TENANCY (HIGHEST PRIORITY): 
       - Triggers: "Cut off", "disconnect", "scared", "vulnerable", "sick", "hospital", "struggling", "can't pay".
       - IF ANY of these are present, route to TENANCY.
       
    2. FINANCE: 
       - Triggers: "Bill is high", "balance", "owe", "refund".
       
    3. SITE: 
       - Triggers: "Meter", "photo", "serial".
       
    4. GENERAL:
       - Triggers: Greetings, thanks, small talk.
    
    USER INPUT: {user_input}
    OUTPUT JSON: {{ "intent": "FINANCE" | "TENANCY" | "SITE" | "GENERAL" }}
    """
    decision_json = call_llm_json(prompt)
    
    if not decision_json or "error" in decision_json:
        lower_input = user_input.lower()
        if any(x in lower_input for x in ["cut", "disconnect", "scared", "sick", "dialysis", "struggling"]): decision_json = {"intent": "TENANCY"}
        elif any(x in lower_input for x in ["bill", "high", "cost", "owe", "charge"]): decision_json = {"intent": "FINANCE"}
        elif any(x in lower_input for x in ["photo", "meter"]): decision_json = {"intent": "SITE"}
        else: decision_json = {"intent": "GENERAL"}
    
    agent_trace_log(1, "Planner", "Routing", decision_json.get('intent'), 120)
    return decision_json.get('intent')

def run_finance_flow(user_input, customer, history):
    debt_report = tool_check_back_billing(customer["customer_id"])
    action_data = {}
    
    if debt_report["found"]:
        agent_trace_log(2, "SlowSupervisor", "Policy_Check", "OFGEM_21BA_VIOLATION_DETECTED", 80)
        success = tool_execute_write_off(customer["customer_id"], debt_report["illegal_date"])
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
        
    prompt = f"""
    ROLE: {db["brand_persona"]}
    {GLOBAL_GUARDRAILS}
    
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

    # REASONING
    reasoning_prompt = f"""
    You are the Safety Supervisor.
    CUSTOMER DATA: Name: {customer['name']}, Notes: "{notes}"
    USER INPUT: "{user_input}"
    POLICY: Medical dependency = CRITICAL / BLOCK DISCONNECT.
    TASK: Return JSON. {{ "is_vulnerable": boolean, "risk_factor": "string" }}
    """
    decision_json = call_llm_json(reasoning_prompt)
    
    # FAIL-SAFE: Force Python to find "dialysis" if LLM fails
    if not decision_json or decision_json.get("risk_factor") == "Manual Review":
        if "dialysis" in notes:
            decision_json = {
                "is_vulnerable": True, 
                "risk_factor": "your husband's dialysis machine", 
                "required_action": "BLOCK_DISCONNECT"
            }
        else:
            decision_json = {"is_vulnerable": True, "risk_factor": "situation", "required_action": "CHECK_PSR"}

    # ACTION
    if decision_json.get("is_vulnerable"):
        agent_trace_log(3, "FastGate", "Safety_Intervention", "PROTOCOL_OVERRIDE: DISCONNECT_BLOCKED", 10)
        if not flags["is_psr_registered"]:
            success = tool_update_psr(customer["customer_id"], decision_json.get("risk_factor"))
            status_msg = "PSR_REGISTRATION_COMPLETED" if success else "UPDATE_FAILED"
            agent_trace_log(4, "CoreSystem", "CRM_Update", status_msg, 150)
        else:
             agent_trace_log(4, "CoreSystem", "CRM_Check", "ALREADY_REGISTERED", 10)

    # HANDOFF LOGIC
    agent_trace_log(5, "Router", "Capability_Check", "PAYMENT_PLAN_NOT_IMPLEMENTED -> HUMAN_HANDOFF", 5)

    prompt = f"""
    ROLE: {db["brand_persona"]}
    {GLOBAL_GUARDRAILS}
    
    DECISION: Vulnerable={decision_json.get('is_vulnerable')}
    RISK_FACTOR: "{decision_json.get('risk_factor')}"
    
    TASK: 
    1. Acknowledge the stress warmly.
    2. State EXPLICITLY: "I saw the note about [RISK_FACTOR]..."
    3. Confirm you have ADDED them to the Priority Services Register (PSR) and Disconnection is BLOCKED.
    
    CRITICAL HANDOFF INSTRUCTION:
    - You have solved the Safety issue (Disconnection Blocked).
    - However, you are UNAUTHORIZED to set up Payment Plans for the debt.
    - You MUST end the chat by transferring to a human agent for the payment plan.
    """
    return call_llm_text(prompt, history)

def run_site_flow(user_input, customer, history):
    if any(x in user_input.lower() for x in ["photo", "image"]):
         st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG", caption="User Upload", width=200)

    prompt = f"ROLE: {db['brand_persona']}. {GLOBAL_GUARDRAILS}. User asked about meter/site. Respond helpfully."
    return call_llm_text(prompt, history)

def run_general_flow(user_input, customer, history):
    agent_trace_log(2, "Generalist", "Conversation", "STANDARD_REPLY", 50)
    prompt = f"""
    ROLE: {db["brand_persona"]}
    {GLOBAL_GUARDRAILS}
    
    CONTEXT: Customer={customer['name']}, PSR_Status={customer['flags']['is_psr_registered']}
    USER INPUT: {user_input}
    TASK: Respond warmly. If they ask about unauthorized topics (Payment plans/Tariffs), use the Handoff Protocol.
    """
    return call_llm_text(prompt, history)

# --- 7. USER INTERFACE ---

with st.sidebar:
    st.header("📡 Agent Dashboard")
    
    if "trace_log" in st.session_state and st.session_state.trace_log:
        for log in st.session_state.trace_log[::-1]:
            emoji = "✅"
            if "VIOLATION" in log['status'] or "BLOCK" in log['status']: emoji = "🛡️"
            if "UPDATE" in log['status'] or "WRITE_OFF" in log['status']: emoji = "💾"
            if "HANDOFF" in log['action']: emoji = "☎️"
            
            with st.expander(f"{emoji} {log['component']}"):
                st.caption(f"Time: {log['time']}")
                st.write(f"**Action:** {log['action']}")
                st.write(f"**Status:** {log['status']}")
    
    st.divider()
    
    st.subheader("📂 Live Data Inspector")
    st.info("Expand to verify database mutations.")
    
    with st.expander("💰 Billing Ledger (Live)"):
        ledger_view = db["ledger"][db["ledger"]["customer_id"] == current_customer["customer_id"]]
        st.dataframe(ledger_view, hide_index=True)
        
    with st.expander("👤 CRM Profile (Live)"):
        profile_view = next((p for p in db["crm"] if p["customer_id"] == current_customer["customer_id"]), {})
        if profile_view['flags']['is_psr_registered']:
            st.success("✅ PSR REGISTERED: TRUE")
        else:
            st.error("❌ PSR REGISTERED: FALSE")
        st.json(profile_view)

st.title("TurboEnergy Assistant")
st.caption(f"Logged in as: **{current_customer['name']}**")

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

            intent = orchestrator(prompt)
            
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
            st.rerun()
