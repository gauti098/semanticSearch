import csv, torch,whisper,os
from moviepy.video.io.VideoFileClip import VideoFileClip
from datetime import datetime, timedelta
# from service.live_stream_encoding_service import LiveStreamEncodeService
 

def format_seconds(seconds):
    # Create a timedelta object with the given number of seconds
    delta = timedelta(seconds=seconds)
 
    # Use the timedelta to create a datetime object starting from a zero point
    base_time = datetime(1, 1, 1, 0, 0, 0)
    result_time = base_time + delta
 
    # Format the datetime object as a string
    formatted_time = result_time.strftime("%H:%M:%S.%f")[:-5] # Remove microseconds
 
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
 
def transcript(self):
    torch.cuda.is_available()
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    # Load the Whisper model:
    # model = whisper.load_model("base", device=DEVICE)
    # model = whisper.load_model("small", device=DEVICE)
    print("loading model-----------------------------------------------")
    model1 = whisper.load_model("small", device=DEVICE,download_root=os.path.join("/home/gautam/Documents/wspace/models/", "small.pt"))
    transcription = model1.transcribe(self)
    print("\n")
    print("transcription text is-----------------------------", transcription["text"])
 
    csv_file_path = "subtitle.csv"
    data = []
    d1 =[]
    d1.append("id")
    d1.append("text")
    d1.append("start")
    d1.append("end")
    data.append(d1)
    segements = transcription["segments"]
 
    print("segments of transcription--------------------------------------")
    for segment in segements:
        data1 = []
        data1.append(segment["id"])
        data1.append(segment["text"])
        data1.append(format_seconds(segment["start"]))
        data1.append(format_seconds(segment["end"]))
        data.append(data1)
    print("---------------------------------")
    # upload_file = LiveStreamEncodeService()
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)
    # Open a text file in write mode ('w')
    # with open("transcript.txt", 'w') as txt_file:
    # # Write each row of data to the text file
    #     for row in data:
    #         # Convert each element in the row to a string and join them with a tab separator
    #         row_str = '\t'.join(map(str, row))
    #         # Write the joined row to the file
    #         txt_file.write(row_str + '\n')
    #     print("generated transcript file----------------------")

    vtt_content = generate_transcript_vtt(transcription)

    vtt_file_path = "transcript.vtt"
    with open(vtt_file_path, 'w') as vtt_file:
        vtt_file.write(vtt_content)
    print("Generated transcript in WebVTT format:", vtt_file_path)
 
    # parent_transcript = "/dash/video_sr2295fab0-72fe-4d36-a63a-7223d5f96600/transcript/filename"
    # parent_summary = "/dash/video_sr2295fab0-72fe-4d36-a63a-7223d5f96600/summary/filename"
 
    # Open a file in write mode ('w')
    with open('summary.txt', 'w') as file:
        # Write content to the file
        file.write(transcription["text"])
        print("generated summary file----------------------")
    # upload_file.upload_to_s3((self, parent_summary, os.path.abspath("summary.txt")))
    print("uploaded summary txt file in s3---------------------------------------")
 
    # upload_file.upload_to_s3((self, parent_transcript, os.path.abspath("transcript.txt")))
    print("uploaded transcript txt file in s3---------------------------------------")
 
 
    return "Uploaded transcript and summary file."
 
 
def convert_mp4_to_mp3(input_file, output_file):
    video_clip = VideoFileClip(input_file)
    audio_clip = video_clip.audio
    print("converting audio to mp3-------------------------------------")
    audio_clip.write_audiofile(output_file)
 
 
input_file_path = "https://00000000"  # Replace with the path to your input MP4 file
output_file_path = "/home/gautam/Downloads/output_audio.mp3"  # Replace with the desired output MP3 file path
# output_file_path = "/video_segmentation/output_audio.mp3"
 
convert_mp4_to_mp3(input_file_path, output_file_path)
 
transcript("/home/gautam/Downloads/output_audio.mp3")