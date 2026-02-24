import os
from dotenv import load_dotenv
from google import genai

# 1. Load environment variables
load_dotenv()

# 2. Initialize the Client
# The SDK automatically detects 'GEMINI_API_KEY' from your environment.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Generate Content
try:
    response = client.models.generate_content(
        model="gemma-3-4b-it", 
        contents="Explain how the new Google GenAI SDK is different in one sentence."
    )
    print(response.text)
    
except Exception as e:
    print(f"Error: {e}")