import requests
import json
import os
from dotenv import load_dotenv

# Load env to get API URL if set, otherwise default to local
load_dotenv()

# You can change this to your Vercel URL if testing production
# API_URL = "http://127.0.0.1:8000" 
API_URL = "https://sentinel-command-backend.vercel.app" 

print(f"🧪 Testing AI Dispatcher at: {API_URL}")
print("-" * 50)

def test_scenario(description):
    print(f"\n📝 Input: '{description}'")
    
    payload = {
        "type": "auto",  # Force Auto-AI mode
        "location": {
            "lat": 40.7128,
            "lon": -74.0060
        },
        "description": description
    }
    
    try:
        response = requests.post(f"{API_URL}/incidents", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Rsp Type:  {data['type'].upper()}")
            print(f"✅ Rsp Desc:  {data['description']}")
            
            if "(Severity:" in data['description']:
                print("🌟 STATUS: AI MODEL IS WORKING! (Severity detected)")
            else:
                print("⚠️ STATUS: FALLBACK MODE (No severity score in description)")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Tip: Make sure uvicorn is running: 'uvicorn main:app --reload'")

# 1. Tricky Case: Medical term (Myocardial Infarction) that keyword logic might miss, but AI knows is Heart Attack
test_scenario("Subject has suffered a myocardial infarction on 5th avenue")

# 2. Tricky Case: "Cat in tree" -> Should be Fire dept (usually) or low priority
test_scenario("A cat is stuck in a high tree and cannot get down")

# 3. Tricky Case: "Blaze" (We added keyword, but let's see AI context)
test_scenario("Massive blaze engulfing the warehouse structure")
