# 🩺 Medical Audio Transcription System

An AI-powered medical speech transcription system that converts doctor–patient conversations into structured, timestamped transcripts with automatic speaker identification.

Built using **OpenAI Whisper**, **PyTorch**, **MFCC-based speaker diarization**, and a **Gradio web interface**.

---

## 🚀 Features

- 🎙️ Automatic speech-to-text transcription using Whisper Small
- 👨‍⚕️👩‍⚕️ Speaker identification (Doctor / Patient)
- ⏱️ Timestamped conversation transcripts
- 🌐 Easy-to-use Gradio web interface
- 📁 Supports multiple audio formats:
  - WAV
  - MP3
  - M4A
  - OGG
- 💻 Works on CPU or GPU

---

## 🏗️ System Architecture

The application follows a four-stage processing pipeline:

```
Audio File Input
        ↓
Whisper Speech Transcription
        ↓
MFCC Feature Extraction
        ↓
KMeans Speaker Clustering
        ↓
Doctor / Patient Labeled Transcript
```

---

## 🧠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| OpenAI Whisper Small | Speech recognition and transcription |
| PyTorch / Torchaudio | Audio processing and MFCC extraction |
| Librosa | Audio loading and preprocessing |
| Scikit-learn | KMeans clustering for speaker diarization |
| Gradio | Web-based user interface |
| NumPy | Numerical operations |

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/medical-audio-transcription.git
cd medical-audio-transcription
```

Install required dependencies:

```bash
pip install openai-whisper torch torchaudio librosa scikit-learn gradio numpy
```

---

## ▶️ Running the Application

Start the Gradio application:

```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:7860
```

---

## 📋 How It Works

### 1. Audio Upload
The user uploads a doctor–patient audio recording through the Gradio interface.

### 2. Speech Transcription
Whisper Small converts the audio into text segments with timestamps.

### 3. Speaker Diarization

Each speech segment is analyzed using MFCC (Mel-Frequency Cepstral Coefficients):

- Audio segments are converted into 40-dimensional MFCC embeddings
- Feature vectors are averaged over time
- KMeans clustering separates the conversation into two speakers
- Speakers are labeled as Doctor and Patient

### 4. Transcript Generation

The final output is generated in a readable format:

```
[00:00:00] Doctor: What brings you in today?
[00:00:15] Patient: I have been experiencing chest pain.
[00:00:24] Doctor: How long has the pain been present?
```

---

## 📊 Example Output Statistics

A sample processed recording produced:

| Metric | Result |
|-------|--------|
| Audio Duration | 448.3 seconds |
| Speech Segments | 160 |
| Doctor Segments | 95 |
| Patient Segments | 64 |
| Unknown Segments | 1 |

---

## ⚙️ Core Modules

### `transcribe_audio()`
- Converts speech into timestamped text using Whisper.
- Supports GPU acceleration when available.

### `get_speaker_embedding()`
- Extracts 40-dimensional MFCC features from each audio segment.

### `assign_speakers()`
- Uses KMeans clustering to classify segments into two speakers.

### `format_transcript()`
- Generates a clean, timestamped transcript.

### `process_medical_audio()`
- Main pipeline function that manages:
  - Audio loading
  - Transcription
  - Speaker identification
  - Transcript generation

---

## 📈 Model Performance

The Whisper-based model was evaluated using Word Error Rate (WER).

| Checkpoint | WER | Status |
|------------|-----|--------|
| Step 250 | 25.87% | Best checkpoint used for final model |
| Step 500 | 36.64% | Overfitting detected |
| Target | <15% | Achievable with more medical training data |

---

## ⚠️ Limitations

Current speaker diarization uses MFCC + KMeans clustering rather than a dedicated speaker recognition model.

Performance may decrease when:
- Speakers have very similar voices
- Audio quality is poor
- Speech segments are extremely short

Future improvements may include advanced models such as SpeechBrain ECAPA or other neural speaker embeddings.

---

## 🛠 Known Issues & Solutions

| Issue | Solution |
|--------|----------|
| SpeechBrain dependency conflicts | Replaced with torchaudio MFCC implementation |
| `k2` module issues on Windows Python 3.12 | Removed SpeechBrain dependency |
| Gradio `show_copy_button` errors | Use compatible Gradio Textbox parameters |
| `webrtcvad` build failures | Avoided resemblyzer dependency |
| Torchaudio backend compatibility issues | Use current torchaudio transform APIs |

---

## 📂 Project Pipeline

The complete development workflow:

1. Audio preprocessing and cleaning  
2. Audio chunking and segmentation  
3. Dataset preparation  
4. Whisper fine-tuning on medical conversations  
5. Model evaluation using WER  
6. Deployment with a Gradio web application  

---

## 💻 Environment

- Operating System: Windows 10 / 11
- Python Version: 3.12
- Framework: Gradio + PyTorch
- Model: Whisper Small

---

## 🔮 Future Improvements

- Improve transcription accuracy with larger medical datasets
- Integrate advanced neural speaker diarization models
- Add support for more medical specialties
- Enable cloud deployment and API access
- Improve processing speed for long recordings

---

## 📄 License

This project is intended for educational and research purposes. Add an appropriate license before public distribution.

---

## 👤 Author

Developed as an AI-powered medical transcription and speaker diarization project using Whisper, PyTorch, and Gradio.
