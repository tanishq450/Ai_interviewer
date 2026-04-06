import requests
import json

base_url = "http://localhost:8000"

print("1. Testing /interview/start...")
try:
    res = requests.post(f"{base_url}/interview/start", json={"user_id": "test_local_script"})
    print(f"Status: {res.status_code}")
    print(res.text[:200])
except Exception as e:
    print("Error:", e)

