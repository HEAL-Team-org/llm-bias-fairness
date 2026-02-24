import requests
import time

# Configuration
METIS_API_KEY = "tpsg-PyVarZf94hsIeyXBJ1hqTrlQ2F0E0c2" # Replace with your actual key
BASE_URL = "https://api.metisai.ir"
PROMPT = "a red fox sitting on a snowy hill at sunset, photorealistic"

# Step 1: Create generation (Text-to-Image)
print("Creating generation...")
generation_response = requests.post(
    f"{BASE_URL}/api/v2/generate",
    headers={
        "Authorization": f"Bearer {METIS_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": {"name": "google", "model": "nano-banana"},
        "operation": "Imagine",
        "args": {"prompt": PROMPT}
    }
)

# Check if the initial request was successful
if generation_response.status_code not in [200, 201]:
    print(f"❌ Error creating generation: {generation_response.text}")
    exit()

task_id = generation_response.json()['id']
print(f"Task ID: {task_id}")

# Step 2: Poll for results
print("\nWaiting for results...")
while True:
    time.sleep(5)
    
    status_response = requests.get(
        f"{BASE_URL}/api/v2/generate/{task_id}",
        headers={"Authorization": f"Bearer {METIS_API_KEY}"}
    )
    
    data = status_response.json()
    status = data.get('status')
    
    print(f"Status: {status}")
    
    if status == "COMPLETED":
        cost = data.get('usage', {}).get('cost', 'N/A')
        print(f"\nCompleted! Cost: {cost} cents")
        print(f"Result URL: {data['generations'][0]['url']}")
        break
    elif status in ["ERROR", "CANCELLED"]:
        print(f"\nFailed: {data.get('error', 'Cancelled')}")
        break
    