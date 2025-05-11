from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import openai
import os
import tempfile
import subprocess
import whisper
from speech_scoring_function import score_speech_general
import parselmouth
from parselmouth.praat import call
from nltk.corpus import cmudict
import nltk
import uuid

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
model = whisper.load_model("base")

nltk.download("cmudict")
pron_dict = cmudict.dict()

session_results = {}

def extract_pitch(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    return round(mean_pitch, 2)

def get_expected_phonemes(text):
    words = text.lower().split()
    expected = {}
    for word in words:
        if word in pron_dict:
            expected[word] = pron_dict[word][0]
        else:
            expected[word] = ["(not found)"]
    return expected

@app.post("/score")
async def score(audio: UploadFile = File(...), reference: str = Form(None)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, audio.filename)
            with open(audio_path, "wb") as f:
                f.write(await audio.read())

            whisper_result = model.transcribe(audio_path)
            hypothesis = whisper_result["text"].strip()
            expected_text = reference.strip() if reference else hypothesis

            transcript_path = os.path.join(tmpdir, "text.txt")
            transcript_line = f"{os.path.splitext(audio.filename)[0]} {expected_text}"
            with open(transcript_path, "w") as f:
                f.write(transcript_line)

            input_dir = os.path.join(tmpdir, "input")
            os.makedirs(input_dir)
            os.rename(audio_path, os.path.join(input_dir, audio.filename))

            output_dir = os.path.join(tmpdir, "aligned")
            subprocess.run([
                "mfa", "align",
                input_dir,
                "english",
                "english",
                output_dir,
                "--clean"
            ], check=True)

            textgrid_path = os.path.join(output_dir, f"{os.path.splitext(audio.filename)[0]}.TextGrid")
            score_result = score_speech_general(
                reference=expected_text,
                hypothesis=hypothesis,
                textgrid_path=textgrid_path
            )

            mean_pitch = extract_pitch(os.path.join(input_dir, audio.filename))
            expected_phonemes = get_expected_phonemes(expected_text)

            session_id = str(uuid.uuid4())
            session_results[session_id] = {
                "reference": expected_text,
                "hypothesis": hypothesis,
                "score_result": score_result,
                "mean_pitch": mean_pitch,
                "expected_phonemes": expected_phonemes
            }

            return JSONResponse(content={"session_id": session_id, "score_result": score_result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/feedback/{session_id}")
async def feedback(session_id: str):
    try:
        session = session_results.get(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        prompt = f"""
You are an expert English pronunciation coach.
Analyze the following speech feedback data and generate helpful, motivating feedback.
Identify what the speaker did well, where they struggled, and give specific advice on how to improve.

Transcript: "{session['hypothesis']}"
Expected: "{session['reference']}"

Pronunciation Score: {session['score_result']['TOEFL-Based Scoring']['Pronunciation Accuracy']}/100
Fluency Score: {session['score_result']['TOEFL-Based Scoring']['Clarity']}/100
Completeness Score: {session['score_result']['TOEFL-Based Scoring']['Completeness']}/100
Mean Pitch: {session['mean_pitch']} Hz

Phoneme Issues:
{session['score_result']['Phoneme-Level Scoring']}

Expected Phonemes:
{session['expected_phonemes']}

Output Format:
- Summary of overall performance (1 paragraph)
- List of specific pronunciation problems and what to practice
- Prosody suggestions (intonation, pitch)
- Encouraging closing sentence
"""

        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful, friendly English pronunciation coach."},
                {"role": "user", "content": prompt}
            ]
        )

        return JSONResponse(content={
            "gpt_feedback": gpt_response.choices[0].message["content"]
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
