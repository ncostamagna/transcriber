import requests
from dotenv import load_dotenv
import os

load_dotenv()

OUTPUT   = os.getenv("OUTPUT", "transcripcion_hablantes")

def summarize(text):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:12b",
            "options": {
                "num_ctx": 128000
            },
            "prompt": f"""Analyze this transcript and provide:

1. SUMMARY (3-5 sentences)
2. KEY POINTS (bullet points)
3. ACTION ITEMS (if any)
4. IMPORTANT DECISIONS (if any)

TRANSCRIPT:
{text}""",
            "stream": False
        }
    )
    return response.json()["response"]

# Read the transcription we just created
with open(f"{OUTPUT}.txt", "r", encoding="utf-8") as f:
    transcription = f.read()

# Generate summary
print("🧠 Summarizing...")
summary = summarize(transcription)

# Save summary
with open(f"{OUTPUT}_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(f"✅ Summary saved → {OUTPUT}_summary-2.txt")