from ai_agent import EmergencyAgent
from datetime import datetime

def test_agent():
    print("--- Starting AI Agent Test ---")
    try:
        agent = EmergencyAgent()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("Triggering mock fall event...")
        agent.handle_fall_event(timestamp=current_time, is_fast_fall=True)
        print("--- Test Completed Successfully ---")
    except Exception as e:
        print(f"--- Test Failed ---: {e}")

if __name__ == "__main__":
    test_agent()
