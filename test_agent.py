from alert_system import AlertSystem

if __name__ == "__main__":
    print("Testing Fall Detection AI Agent Integration...")
    alert = AlertSystem()
    
    # Send mock alert logic (Fast fall detected)
    alert.send_alert(is_fast_fall=True)
