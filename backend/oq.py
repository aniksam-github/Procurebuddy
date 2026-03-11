from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Hardcode temporarily if env not loading
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in environment")
    exit()

client = Groq(api_key=api_key)

models = client.models.list()

print("Available Groq Models:\n")

for m in models.data:
    print("-", m.id)