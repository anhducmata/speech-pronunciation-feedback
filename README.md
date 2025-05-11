# 🗣️ English Pronunciation Scoring & Feedback API

This project is a FastAPI-based web service that analyzes English speech recordings and returns detailed pronunciation feedback, combining speech recognition, phoneme alignment, acoustic feature extraction, and natural language generation using GPT-4o.

## 🚀 Features

| Feature                      | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| 🎙️ Whisper Transcription     | Automatically converts speech to text using OpenAI Whisper                  |
| 📊 TOEFL-Style Scoring       | Computes Pronunciation, Fluency, and Completeness scores                    |
| 🔤 Phoneme Alignment         | Uses Montreal Forced Aligner (MFA) to match audio to phonemes               |
| 🧠 CMU Dictionary Lookup      | Shows expected phonemes for each word using CMUdict                         |
| 📈 Prosody Analysis           | Extracts average pitch (mean fundamental frequency) with Praat              |
| 🤖 GPT-4o Feedback            | Returns friendly, personalized feedback on how to improve pronunciation     |

## 🔧 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/anhducmata/speech-pronunciation-feedback.git
cd speech-pronunciation-feedback
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Make sure you also install:
```bash
sudo apt install montreal-forced-aligner
```

### 3. Set Environment Variable
```bash
export OPENAI_API_KEY=sk-xxxxxxx
```

## 🎯 API Endpoint

### POST /score-with-feedback

**Form-data Parameters:**
- `audio`: `.wav` audio file (required)
- `reference`: expected sentence (optional)

**Returns:**
- TOEFL-style speech scores
- Misaligned phoneme analysis
- Expected phonemes
- Mean pitch of speech
- GPT-generated improvement feedback

## 🧠 Credits
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict)
- [Praat / Parselmouth](https://parselmouth.readthedocs.io/en/latest/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📜 License
MIT License