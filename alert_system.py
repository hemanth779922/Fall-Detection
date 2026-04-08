import datetime
import config

try:
    # Try importing Twilio (for sending real SMS)
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

class AlertSystem:
    def __init__(self):
        # Keep track of the last alert sent to prevent spamming SMS every second!
        self.last_alert_time = None
        self.cooldown_seconds = 15  # Reduced to 15s to make testing much faster

    def send_alert(self, is_fast_fall=True):
        """
        Sends an SOS alert (Mock or Real) when a fall is detected.
        Checks cooldown to prevent duplicate alerts for the same fall.
        """
        now = datetime.datetime.now()
        
        # Check cooldown to prevent spamming alerts
        if self.last_alert_time is not None:
            time_diff = (now - self.last_alert_time).total_seconds()
            if time_diff < self.cooldown_seconds:
                return  # Skip sending; still in cooldown
        
        # Record the time the alert was sent
        self.last_alert_time = now
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # The user specifically requested these exact concepts
        if is_fast_fall:
            message_body = f"Emergency! Person is falling down fast at {time_str}. He is having an emergency!"
        else:
            message_body = f"Alert! Person is slowly falling a sleep at {time_str}."

        # Print alert clearly to the terminal logger
        print("\n" + "="*50)
        print("!!! INITIATING SOS ALERT !!!")
        print(message_body)
        print("="*50 + "\n")

        # Step 0: Check if n8n is enabled and send Webhook Alert
        if getattr(config, 'USE_N8N', False):
            print(f"[Alert System] Sending Webhook to n8n: {config.N8N_WEBHOOK_URL}...")
            import requests # Make sure requests is installed
            try:
                payload = {
                    "event": "fall_detected",
                    "timestamp": time_str,
                    "message": message_body,
                    "target_phone": config.ALERT_PHONE_NUMBER,
                    "camera_index": config.CAMERA_INDEX
                }
                response = requests.post(config.N8N_WEBHOOK_URL, json=payload)
                if response.status_code == 200:
                    print("[Alert System] n8n Webhook triggered successfully!")
                else:
                    print(f"[Alert System] Failed to trigger n8n Webhook. Status code: {response.status_code}")
            except Exception as e:
                print(f"[Alert System] Error sending to n8n: {e}")

        # Step 0.5: Trigger the LangChain AI Agent integration
        if getattr(config, 'USE_AI_AGENT', False):
            print("\n[Alert System] Routing event to AI Agent Workflow...")
            try:
                import threading
                from ai_agent import EmergencyAgent
                
                def bg_agent_task():
                    try:
                        agent = EmergencyAgent()
                        agent.handle_fall_event(timestamp=time_str, is_fast_fall=is_fast_fall)
                        print("[Alert System] AI Agent background workflow completed.")
                    except Exception as e:
                        print(f"[Alert System Background] Error: {e}")
                
                # Start AI Agent in background thread so the video feed doesn't freeze!
                threading.Thread(target=bg_agent_task, daemon=True).start()
                
                return # Skip the standard hardcoded SMS since the AI is handling it
            except ImportError as e:
                print(f"[Alert System] Error: ai_agent module not found: {e}")
            except Exception as e:
                print(f"[Alert System] Error executing AI Agent: {e}")

        # Step 1: Check configurations over whether to send Real SMS or Mock
        if config.SEND_REAL_SMS:
            if TWILIO_AVAILABLE:
                try:
                    # Authenticate Twilio REST API Client
                    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
                    
                    # Create and send SMS
                    message = client.messages.create(
                        body=message_body,
                        from_=config.TWILIO_FROM_NUMBER,
                        to=config.ALERT_PHONE_NUMBER
                    )
                    print(f"[Alert System] Real SMS Sent successfully! ID: {message.sid}")
                except Exception as e:
                    print(f"[Alert System] Failed to send real SMS: {e}")
            else:
                print("[Alert System] Twilio library not found! Falling back to Mock SMS.")
                print("Hint: Install twilio via: pip install twilio")
                self.mock_sms(message_body)
        else:
            # Fallback to Mock SMS System when Real SMS is disabled
            self.mock_sms(message_body)

    def mock_sms(self, msg):
        """Simulates sending an SMS alert without using external services (cost-free method for academics)."""
        print(f"[MOCK SMS] Sending to Database/Gateway: {config.ALERT_PHONE_NUMBER}")
        print(f"[MOCK SMS] Message Payload: '{msg}'")
        print("[MOCK SMS] Alert Successfully Delivered!\n")
