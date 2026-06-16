# ==========================================================
# MEDICAL AUDIO TRANSCRIPTION
# Fixed: Uses torchaudio only — no SpeechBrain, no resemblyzer
# Works on Python 3.12 + Windows — No C++ build tools needed
# ==========================================================

import gc
import time
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import gradio as gr
import whisper

from sklearn.cluster import KMeans

# ==========================================================
# CONFIGURATION
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading Whisper Small...")
whisper_model = whisper.load_model("small")
print("Whisper loaded successfully")

# ==========================================================
# SPEAKER EMBEDDING USING MFCC (no extra library needed)
# ==========================================================

def get_speaker_embedding(audio_array, start_sec, end_sec, sr=16000):
    try:
        start   = int(start_sec * sr)
        end     = int(end_sec   * sr)
        segment = audio_array[start:end]

        # Skip very short segments
        if len(segment) < 1600:
            return None

        # Convert to tensor
        waveform = torch.tensor(segment).unsqueeze(0).float()

        # Extract MFCC features as speaker embedding
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=40,
            melkwargs={
                "n_fft":    512,
                "n_mels":   40,
                "hop_length": 160
            }
        )

        mfcc       = mfcc_transform(waveform)        # (1, 40, time)
        embedding  = mfcc.mean(dim=2).squeeze().numpy()  # (40,)

        return embedding

    except Exception:
        return None

# ==========================================================
# TRANSCRIPTION
# ==========================================================

def transcribe_audio(audio_array):
    result = whisper_model.transcribe(
        audio_array.astype(np.float32),
        language="en",
        word_timestamps=True,
        fp16=torch.cuda.is_available()
    )
    return result["segments"]

# ==========================================================
# SPEAKER CLUSTERING
# ==========================================================

def assign_speakers(segments, audio_array, sr=16000):
    embeddings = []
    indexes    = []

    for i, seg in enumerate(segments):
        embedding = get_speaker_embedding(
            audio_array,
            seg["start"],
            seg["end"],
            sr
        )
        if embedding is not None:
            embeddings.append(embedding)
            indexes.append(i)

    if len(embeddings) < 2:
        return {
            i: "Doctor"
            for i in range(len(segments))
        }

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(np.array(embeddings))

    speaker_map = {}
    results     = {}

    for index, label in zip(indexes, labels):
        if label not in speaker_map:
            speaker_map[label] = (
                "Doctor"
                if len(speaker_map) == 0
                else "Patient"
            )
        results[index] = speaker_map[label]

    return results

# ==========================================================
# FORMAT TRANSCRIPT
# ==========================================================

def format_transcript(segments, labels):
    output = []

    for i, segment in enumerate(segments):
        speaker   = labels.get(i, "Unknown")
        timestamp = time.strftime(
            "%H:%M:%S",
            time.gmtime(segment["start"])
        )
        text = segment["text"].strip()
        output.append(f"[{timestamp}] {speaker}: {text}")

    return "\n".join(output)

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def process_medical_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file", ""

    try:
        print("Loading audio...")
        audio_array, sr = librosa.load(
            audio_file,
            sr=16000,
            mono=True
        )
        duration = len(audio_array) / sr

        print("Transcribing audio...")
        segments = transcribe_audio(audio_array)

        if not segments:
            return "No speech detected", ""

        print("Detecting speakers...")
        speaker_labels = assign_speakers(
            segments,
            audio_array,
            sr
        )

        transcript = format_transcript(
            segments,
            speaker_labels
        )

        doctor_count = sum(
            1 for value in speaker_labels.values()
            if value == "Doctor"
        )
        patient_count = sum(
            1 for value in speaker_labels.values()
            if value == "Patient"
        )

        summary = (
            f"Completed\n\n"
            f"Duration        : {duration:.1f} seconds\n"
            f"Segments        : {len(segments)}\n"
            f"Doctor segments : {doctor_count}\n"
            f"Patient segments: {patient_count}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return summary, transcript

    except Exception as error:
        return f"Error: {str(error)}", ""

# ==========================================================
# GRADIO USER INTERFACE
# ==========================================================

with gr.Blocks(title="Medical Audio Transcription") as app:

    gr.Markdown("""
    # Medical Audio Transcription
    Upload a doctor-patient conversation and receive a speaker-labeled transcript.
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Upload WAV or MP3 File",
                type="filepath"
            )
            run_button = gr.Button(
                "Transcribe",
                variant="primary"
            )

        with gr.Column():
            summary_output = gr.Textbox(
                label="Summary",
                lines=6
            )

    transcript_output = gr.Textbox(
        label="Doctor / Patient Transcript",
        lines=20
    )

    run_button.click(
        fn=process_medical_audio,
        inputs=audio_input,
        outputs=[
            summary_output,
            transcript_output
        ]
    )

# ==========================================================
# START APPLICATION
# ==========================================================

print("Starting Gradio application...")
app.launch(
    share=False,
    debug=False
)