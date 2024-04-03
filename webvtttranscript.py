import csv
import torch
import whisper
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from datetime import datetime, timedelta

import speech_recognition as sr
from pydub import AudioSegment


def format_seconds(seconds):
    delta = timedelta(seconds=seconds)
    base_time = datetime(1, 1, 1, 0, 0, 0)
    result_time = base_time + delta
    formatted_time = result_time.strftime("%H:%M:%S.%f")[:-5]
    return formatted_time

def generate_transcript_vtt(transcription):
    vtt_content = "WEBVTT\n\n"
    segments = transcription["segments"]
    for segment in segments:
        start_time = format_seconds(segment["start"])
        end_time = format_seconds(segment["end"])
        text = segment["text"]
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt_content

def transcript(audio_file_path):
    DEVICE = "cpu"
    model1 = whisper.load_model("small", device=DEVICE, download_root=os.path.join("/home/gautam/Documents/wspace/models/", "small.pt"))
    transcription = model1.transcribe(audio_file_path)

    vtt_content = generate_transcript_vtt(transcription)

    vtt_file_path = "transcript.vtt"
    with open(vtt_file_path, 'w') as vtt_file:
        vtt_file.write(vtt_content)
    print("Generated transcript in WebVTT format:", vtt_file_path)

    return "Uploaded transcript file."

def convert_mp4_to_mp3(input_file, output_file):
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    print("Converting audio to mp3...")
    audio_clip.write_audiofile(output_file)

input_file_path = "/home/gautam/Downloads/video_sr2295fab0-72fe-4d36-a63a-7223d5f96600.mp4"
output_file_path = "/home/gautam/Downloads/output_audio.mp3"

def detect_language(audio_file_path):
    # Convert MP3 to WAV
    wav_file_path = audio_file_path[:-4] + ".wav"  # Change file extension to WAV
    audio = AudioSegment.from_mp3(audio_file_path)
    audio.export(wav_file_path, format="wav")

    # Detect language from WAV
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)  # Read the entire audio file
        detected_language = recognizer.recognize_google(audio_data, show_all=True)
        if "language" in detected_language:
            detected_language = detected_language["language"]
        else:
            detected_language = "Language not detected"

    # Delete temporary WAV file
    os.remove(wav_file_path)

    return detected_language

output_language = detect_language(output_file_path)
print("Detected language:", output_language)

convert_mp4_to_mp3(input_file_path, output_file_path)
transcript(output_file_path)
