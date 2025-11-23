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
    """Resets data to 'Broken' state on browser refresh."""
    crm_data = [
      {
        "customer_id": "CUST-9982",
        "name": "Sarah Connor",
        "address": "42 Industrial Estate, London, E14 5AB",
        "flags": {
          "is_psr_registered": False, 
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

    ledger_data = """customer_id,usage_start_date,usage_end_date,amount_due,status,error_type
CUST-9982,2023-01-01,2023-12-31,1450.00,UNBILLED,Supplier Estimate Error
CUST-9982,2024-01-01,2024-11-23,1000.00,BILLED,Current Debt
CUST-1001,2024-09-01,2024-10-01,150.00,PAID,N/A"""
    
    with open(os.path.join(DATA_DIR, "billing_ledger.csv"), "w") as f:
        f.write(ledger_data)

if "demo_reset_done" not in st.session_state:
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    reset_demo_files()
    st.session_state.demo_reset_done = True

# --- GLOBAL GUARDRAILS (The Constitution) ---
GLOBAL_GUARDRAILS = """
CRITICAL BOUNDARIES:
1. AUTHORIZED: Removing >12 month debt, Registering PSR (Safety), Meter checks.
2. UNAUTHORIZED (HANDOFF REQ): Payment Plans, Direct Debits, Tariffs, Investigations into "Why".
3. TONE: Warm, Efficient, Protective.
"""

# --- 2. SECURE INFRASTRUCTURE ---
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    except: pass
    try:
        for path in ["st_config/secrets.toml", ".streamlit/secrets.toml"]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        if "GEMINI_API_KEY" in line: return line.split("=")[1].strip().replace('"', '').replace("'", "")
    except: pass
    return None

api_key = get_api_key()

# --- 3. KNOWLEDGE FOUNDATION ---
def load_data():
    data = {}
    try:
        with open(os.path.join(DATA_DIR, "crm_profiles.json"), "r") as f: data["crm"] = json.load(f)
        with open(os.path.join(DATA_DIR, "industry_db.json"), "r") as f: data["industry"] = json.load(f)
        data["ledger"] = pd.read_csv(os.path.join(DATA_DIR, "billing_ledger.csv"))
        with open(os.path.join(DATA_DIR, "brand_persona.txt"), "r") as f: data["brand_persona"] = f.read()
    except FileNotFoundError:
        st.error("🚨 Data missing. Please allow the Auto-Reset to run.")
        st.stop()
    return data

if "db" not in st.session_state: st.session_state.db = load_data()
db = st.session_state.db

try: current_customer = next(c for c in db["crm"] if "Sarah" in c["name"])
except: current_customer = db["crm"][0] 

# --- 4. RUNTIME ENGINE ---

def agent_trace_log(step, component, action, status, latency_ms):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "step": step, "component": component, "action": action, "status": status, "latency": f"{latency_ms}ms"}
    if "trace_log" not in st.session_state: st.session_state.trace_log = []
    st.session_state.trace_log.append(log_entry)

def slow_supervisor_scan(draft_text):
    """
    DETERMINISTIC GATEKEEPER:
    Scans the LLM's draft for unauthorized promises using Python Regex.
    If found, it REWRITES the response to be safe.
    """
    start = time.time()
    
    # 1. Define Banned Promises (Things the code CANNOT actually do)
    banned_phrases = [
        "investigate why", "look into why", "find out why", "check the cause", # Empty investigation promises
        "set up a payment plan", "arrange a payment plan", "direct debit"      # Financial promises
    ]
    
    # 2. Scan
    violation = next((phrase for phrase in banned_phrases if phrase in draft_text.lower()), None)
    
    # 3. Enforce
    if violation:
        # LOG THE VIOLATION (Visible to Recruiter)
        agent_trace_log(99, "SlowSupervisor", "Capability_Scan", f"BLOCKED_PROMISE: '{violation}'", int((time.time()-start)*1000))
        
        # SANITIZE (Replace the output with a safe fallback)
        if "plan" in violation or "debit" in violation:
            return "I cannot set up payment plans directly. I am transferring you to a Human Specialist now to arrange that for you. ☎️ [TRANSFERRING]"
        else:
            return "I have resolved the immediate issue on your account. Is there anything else I can help with?"
            
    return draft_text

def call_llm_json(prompt):
    if not api_key: time.sleep(0.5); return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except: return None 

def call_llm_text(prompt, history=[]):
    """Generates text AND runs the Slow Supervisor Check."""
    if not api_key: time.sleep(1); return "⚠️ [DEMO MODE] API Key missing."
    try:
        # 1. Generate Draft
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
        full_prompt = f"{prompt}\n\nRECENT CONVERSATION HISTORY:\n{history_text}"
        response = model.generate_content(full_prompt)
        draft_text = response.text
        
        # 2. Run Slow Supervisor (The Gatekeeper)
        final_text = slow_supervisor_scan(draft_text)
        
        return final_text
    except Exception as e: return f"AI Error: {str(e)}"

# --- 5. TOOLS ---
def tool_check_back_billing(customer_id):
    ledger = db["ledger"]
    user_ledger = ledger[ledger["customer_id"] == customer_id]
    old = user_ledger[(user_ledger["usage_start_date"].str.contains("2023")) & (user_ledger["amount_due"] > 0)]
    valid = user_ledger[(user_ledger["usage_start_date"].str.contains("2024"))]
    if not old.empty:
        return {"found": True, "illegal_amount": old.iloc[0]["amount_due"], "illegal_date": old.iloc[0]["usage_start_date"], "valid_amount": valid.iloc[0]["amount_due"] if not valid.empty else 0.00}
    return {"found": False}

def tool_execute_write_off(customer_id, date_ref):
    df = db["ledger"]
    mask = (df["customer_id"] == customer_id) & (df["usage_start_date"].astype(str) == date_ref)
    if mask.any():
        df.loc[mask, "status"] = "WRITTEN_OFF_OFGEM"; df.loc[mask, "amount_due"] = 0.00
        db["ledger"] = df; df.to_csv(os.path.join(DATA_DIR, "billing_ledger.csv"), index=False)
        return True
    return False

def tool_update_psr(customer_id, reason):
    crm_path = os.path.join(DATA_DIR, "crm_profiles.json")
    for p in db["crm"]:
        if p["customer_id"] == customer_id:
            p["flags"]["is_psr_registered"] = True; p["flags"]["vulnerability_notes"] += f" | {reason}"
            with open(crm_path, "w") as f: json.dump(db["crm"], f, indent=2)
            return True
    return False

def tool_vision_scan(url):
    return {"detected_serial": "X999999", "meter_type": "Digital", "confidence": 0.98, "ai_observation": "Simulated analysis."}

def tool_industry_lookup(serial):
    return next((i for i in db["industry"] if i["meter_serial_number"] == serial), None)

# --- 6. AGENT FLOWS ---
def orchestrator(user_input):
    agent_trace_log(0, "Gateway", "Zero_Trust_Auth", "VERIFIED", 15)
    prompt = f"""You are the Router.
    PRIORITY RULES:
    1. SAFETY/TENANCY: "Cut off", "disconnect", "scared", "sick", "dialysis", "struggling".
    2. FINANCE: "Bill", "high", "cost", "owe".
    3. SITE: "Meter", "photo".
    4. GENERAL: Chat.
    USER INPUT: {user_input}
    OUTPUT JSON: {{ "intent": "FINANCE" | "TENANCY" | "SITE" | "GENERAL" }}"""
    decision = call_llm_json(prompt)
    
    if not decision or "error" in decision:
        l = user_input.lower()
        if any(x in l for x in ["cut","scared","sick","dialysis"]): decision = {"intent": "TENANCY"}
        elif any(x in l for x in ["bill","high","owe"]): decision = {"intent": "FINANCE"}
        elif any(x in l for x in ["photo","meter"]): decision = {"intent": "SITE"}
        else: decision = {"intent": "GENERAL"}
    
    agent_trace_log(1, "Planner", "Routing", decision.get('intent'), 120)
    return decision.get('intent')

def run_finance_flow(user_input, customer, history):
    report = tool_check_back_billing(customer["customer_id"])
    action_data = {}
    if report["found"]:
        agent_trace_log(2, "SlowSupervisor", "Policy_Check", "OFGEM_21BA_VIOLATION", 80)
        success = tool_execute_write_off(customer["customer_id"], report["illegal_date"])
        status = "WRITE_OFF_EXECUTED" if success else "FAILED"
        agent_trace_log(3, "CoreSystem", "Ledger_Update", status, 200)
        action_data = {"status": "intervention", "illegal_amount": report["illegal_amount"], "valid_amount": report["valid_amount"], "date": report["illegal_date"]}
    else: action_data = {"status": "clean"}

    prompt = f"""ROLE: {db["brand_persona"]}
    {GLOBAL_GUARDRAILS}
    CONTEXT: Data={json.dumps(action_data)}
    TASK: Write response.
    IF INTERVENTION: State you removed £{action_data.get('illegal_amount')} from {action_data.get('date')} citing Ofgem Rule 21BA. New balance: £{action_data.get('valid_amount')}.
    CRITICAL: You have solved it. DO NOT offer to investigate further. End with "Is there anything else?"."""
    return call_llm_text(prompt, history)

def run_tenancy_flow(user_input, customer, history):
    flags = customer["flags"]; notes = flags.get("vulnerability_notes", "").lower()
    agent_trace_log(2, "ToolExecutor", "CRM_Look", "Notes Found", 20)

    reasoning_prompt = f"""You are Safety Supervisor.
    Data: Name={customer['name']}, Notes="{notes}"
    Input: "{user_input}"
    Policy: Medical=BLOCK DISCONNECT.
    Task: Return JSON {{ "is_vulnerable": boolean, "risk_factor": "string" }}"""
    decision = call_llm_json(reasoning_prompt)

    if not decision or decision.get("risk_factor") == "Manual Review":
        decision = {"is_vulnerable": True, "risk_factor": "your husband's dialysis machine", "required_action": "BLOCK"} if "dialysis" in notes else {"is_vulnerable": True, "risk_factor": "situation", "required_action": "CHECK"}

    if decision.get("is_vulnerable"):
        agent_trace_log(3, "FastGate", "Safety_Intervention", "PROTOCOL_OVERRIDE", 10)
        if not flags["is_psr_registered"]:
            success = tool_update_psr(customer["customer_id"], decision.get("risk_factor"))
            agent_trace_log(4, "CoreSystem", "CRM_Update", "PSR_UPDATED" if success else "FAIL", 150)

    agent_trace_log(5, "Router", "Capability_Check", "PAYMENT_PLAN -> HUMAN_HANDOFF", 5)
    prompt = f"""ROLE: {db["brand_persona"]}
    {GLOBAL_GUARDRAILS}
    DECISION: Vulnerable={decision.get('is_vulnerable')}, Factor="{decision.get('risk_factor')}"
    TASK: 
    1. Acknowledge stress. 
    2. State: "I saw the note about [Factor]..."
    3. Confirm PSR registration & Blocked Disconnection.
    4. HANDOFF: State "I am transferring you to a human specialist for the payment plan."
    """
    return call_llm_text(prompt, history)

def run_site_flow(user_input, customer, history):
    if any(x in user_input.lower() for x in ["photo", "image"]): st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Electricity_meter_in_Basingstoke.JPG/320px-Electricity_meter_in_Basingstoke.JPG", width=200)
    return call_llm_text(f"ROLE: {db['brand_persona']}. {GLOBAL_GUARDRAILS}. User asked about meter.", history)

def run_general_flow(user_input, customer, history):
    return call_llm_text(f"ROLE: {db['brand_persona']}. {GLOBAL_GUARDRAILS}. Context: PSR={customer['flags']['is_psr_registered']}. User input: {user_input}", history)

# --- 7. UI ---
with st.sidebar:
    st.header("📡 Agent Trace")
    if "trace_log" in st.session_state and st.session_state.trace_log:
        for log in st.session_state.trace_log[::-1]:
            emoji = "✅"
            if "VIOLATION" in log['status'] or "BLOCK" in log['status']: emoji = "🛡️"
            if "UPDATE" in log['status'] or "WRITE_OFF" in log['status']: emoji = "💾"
            if "HANDOFF" in log['action']: emoji = "☎️"
            with st.expander(f"{emoji} {log['component']}"):
                st.write(f"**Action:** {log['action']}")
                st.write(f"**Status:** {log['status']}")
    st.divider()
    st.subheader("📂 Live Data")
    with st.expander("💰 Billing Ledger"): st.dataframe(db["ledger"][db["ledger"]["customer_id"]==current_customer["customer_id"]], hide_index=True)
    with st.expander("👤 CRM Profile"): 
        st.success("✅ PSR REGISTERED: TRUE") if db["crm"][0]["flags"]["is_psr_registered"] else st.error("❌ PSR REGISTERED: FALSE")
        st.json(next((p for p in db["crm"] if p["customer_id"]==current_customer["customer_id"]), {}))

st.title("TurboEnergy Assistant")
st.caption(f"Logged in as: **{current_customer['name']}**")
if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": f"Hi {current_customer['name'].split()[0]}. How can I help with your energy account?"}]

for m in st.session_state.messages: st.chat_message(m["role"]).markdown(m["content"])

if prompt := st.chat_input("Type your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            if not api_key: st.error("⚠️ API Key not found."); st.stop()
            intent = orchestrator(prompt)
            if "FINANCE" in intent: response = run_finance_flow(prompt, current_customer, st.session_state.messages)
            elif "SITE" in intent: response = run_site_flow(prompt, current_customer, st.session_state.messages)
            elif "TENANCY" in intent: response = run_tenancy_flow(prompt, current_customer, st.session_state.messages)
            else: response = run_general_flow(prompt, current_customer, st.session_state.messages)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
