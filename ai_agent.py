import os
import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# ----------------------------------------------------------------
# 1. AI AGENT TOOLS
# ----------------------------------------------------------------

@tool
def send_sos_alert(recipient: str, message: str) -> str:
    """Useful for sending an emergency SMS or message to a recipient's phone number or ID."""
    print("\n" + "*"*50)
    print("[AI AGENT ACTION] -> send_sos_alert()")
    print(f"To: {recipient}")
    print(f"Message: {message}")
    
    if getattr(config, 'SEND_REAL_SMS', False):
        try:
            from twilio.rest import Client
            client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
            twilio_msg = client.messages.create(
                body=message,
                from_=config.TWILIO_FROM_NUMBER,
                to=config.ALERT_PHONE_NUMBER
            )
            print(f"Twilio SMS dispatched! SID: {twilio_msg.sid}")
        except Exception as e:
            print(f"Twilio error: {e}")
    else:
        print("[MOCK SMS] Real SMS disabled in config.")
        
    print("*"*50 + "\n")
    return "Alert sent successfully to the provided contact."

@tool
def get_patient_emergency_contact(patient_id: str) -> str:
    """Useful for looking up the emergency contact phone number and name for a given patient ID."""
    database = {
        "patient_001": f"Dr. Smith (Primary Contact) - Phone: {config.ALERT_PHONE_NUMBER}",
        "patient_002": "Family Member - Phone: +1234567890"
    }
    contact = database.get(patient_id)
    if contact:
        return f"Found Contact: {contact}"
    return "Unknown Contact."

# ----------------------------------------------------------------
# 2. AGENT INITIALIZATION & EXECUTION
# ----------------------------------------------------------------

class EmergencyAgent:
    def __init__(self):
        # Avoid crashing if config is missing/misconfigured; allow env override.
        self.api_key = getattr(config, "GOOGLE_API_KEY", None) or getattr(config, "ANTHROPIC_API_KEY", None) or os.getenv("GOOGLE_API_KEY", "")
        
        if self.api_key and self.api_key != "your-anthropic-api-key":
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            self.tools = [send_sos_alert, get_patient_emergency_contact]
            # Create a ReAct Agent using modern LangGraph API
            self.agent_executor = create_react_agent(self.llm, tools=self.tools)
        else:
            self.agent_executor = None

    def handle_fall_event(self, timestamp: str, is_fast_fall: bool):
        print("\n[AI AGENT] Checking system keys...")
        if not self.agent_executor:
            print("[AI AGENT] Running in MOCK Mode (No API Key).")
            print(f"\n[MOCK AI AGENT] Waking up to handle fall at {timestamp}...")
            print("[MOCK AI AGENT] Since no API key is provided, generating an automatic simulated response.")
            print("\n[AI AGENT COMPLETION SUMMARY]:")
            print("Simulated Output: Detected Fall Incident. Sending SOS message to emergency contact (Dr. Smith) according to database.")
            
            if getattr(config, 'SEND_REAL_SMS', False):
                try:
                    from twilio.rest import Client
                    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
                    msg_body = f"AI EMERGENCY ALERT: Fall detected at {timestamp}. Immediate assistance required."
                    twilio_msg = client.messages.create(
                        body=msg_body,
                        from_=config.TWILIO_FROM_NUMBER,
                        to=config.ALERT_PHONE_NUMBER
                    )
                    print(f"[MOCK AI AGENT] Twilio SMS dispatched successfully! SID: {twilio_msg.sid}")
                except Exception as e:
                    print(f"[MOCK AI AGENT] Twilio error: {e}")
            else:
                print("[MOCK AI AGENT] Real SMS sending is disabled in config.")
                
            return

        severity = "HIGH SEVERITY (FAST FALL)" if is_fast_fall else "MEDIUM SEVERITY (SLOW FALL / GROUNDED)"
        patient_id = "patient_001"
        
        system_prompt = "You are a highly efficient emergency medical AI. When a fall occurs, your only job is to find the contact info and immediately dispatch an SOS alert using your tools."
        
        prompt_text = f"""
        URGENT EMERGENCY NOTIFICATION:
        Your computer vision model just detected a physical fall.
        
        Context:
        - Time: {timestamp}
        - Severity: {severity}
        - Patient ID: {patient_id}
        
        Task Workflow:
        1. Find the emergency contact for this patient using your tools.
        2. Draft an urgent, professional SOS message explaining the fall and its severity.
        3. Send the SOS alert to the emergency contact you found using the send_sos_alert tool.
        """
        
        print(f"\n[AI AGENT] Waking up to handle fall at {timestamp}...")
        try:
            # LangGraph messages input format
            inputs = {"messages": [("system", system_prompt), ("user", prompt_text)]}
            response = self.agent_executor.invoke(inputs)
            
            # Print the final summarization from the AI
            final_message = response["messages"][-1].content
            print(f"\n[AI AGENT COMPLETION SUMMARY]:\n{final_message}")
            
        except Exception as e:
            print(f"[AI AGENT ERROR] Encountered an error during workflow: {e}")