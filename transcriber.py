from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from datetime import timedelta
import torch
from dotenv import load_dotenv
import os
import subprocess
import tempfile

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
VIDEO    = os.getenv("VIDEO", "video.mp4")
OUTPUT   = os.getenv("OUTPUT", "transcripcion_hablantes")

# Extract audio to WAV to avoid container duration mismatches
_tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
_tmp_wav.close()
AUDIO = _tmp_wav.name
subprocess.run(
    ["ffmpeg", "-y", "-i", VIDEO, "-ac", "1", "-ar", "16000", AUDIO],
    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

# ── 1. Diarization (who speaks when) ───────────────────────
print("🔍 Identifying speakers...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)
output = pipeline(AUDIO)
diarization = output.speaker_diarization

def get_speaker(start, end, diarization):
    best_speaker, best_overlap = "Unknown", 0.0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, speaker
    return best_speaker

# ── 2. Transcription ────────────────────────────────────────
print("📝 Transcribing...")
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
segments, info = model.transcribe(AUDIO, language="en")
segments = list(segments)

# ── 3. Merge and save ───────────────────────────────────────
def fmt_tiempo(seconds):
    return str(timedelta(seconds=int(seconds)))

with open(f"{OUTPUT}.txt", "w", encoding="utf-8") as f:
    current_speaker = None

    for segment in segments:
        speaker = get_speaker(segment.start, segment.end, diarization)

        # Only print speaker name when it changes
        if speaker != current_speaker:
            f.write(f"\n🎤 {speaker} [{fmt_tiempo(segment.start)}]\n")
            current_speaker = speaker

        f.write(f"   {segment.text.strip()}\n")

print(f"✅ Saved to {OUTPUT}.txt")
os.unlink(AUDIO)