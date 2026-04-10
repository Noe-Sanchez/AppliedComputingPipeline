# Overall imports
import pandas as pd
import numpy as np
import pathlib
import math
import os
import cv2
import re

def extract_frames(video_path, output_folder, frame_skip=1):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print(f"Error opening file: {video_path}")
    return

  frame_count = 0
  saved_frames = 0

  print(f"Processing '{video_path}'...")

  while True:
    ret, frame = cap.read()

    if not ret:
      break
   
    if frame_count % frame_skip == 0:
      file_name = os.path.join(output_folder, f"frame_{saved_frames:04d}.jpg")
      
      cv2.imwrite(file_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
      saved_frames += 1

    frame_count += 1

  cap.release()

  print(f"Finished processing '{video_path}'. Total frames: {frame_count}, Saved frames: {saved_frames}.")

def extract_telemetry(srt_file_path):
  gen_regex = r":\s*([\d\.\-]+)" 

  data = pd.DataFrame(columns=['Frame', 'rel_alt', 'latitude', 'longitude'])

  try:
    with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
      telemetry = srt_file.read()

      data.loc[:, 'rel_alt']   = re.findall("rel_alt"+  gen_regex, telemetry)
      data.loc[:, 'latitude']  = re.findall("latitude"+ gen_regex, telemetry)
      data.loc[:, 'longitude'] = re.findall("longitude"+gen_regex, telemetry)
      data.loc[:, 'Frame'] = range(1, len(data) + 1)

      # Make columns numeric
      data['rel_alt']   = pd.to_numeric(data['rel_alt'],   errors='coerce')
      data['latitude']  = pd.to_numeric(data['latitude'],  errors='coerce')
      data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')

    if not data.empty:
      return data
    else:
      print("No 'rel_alt' values found in the SRT file.")

  except FileNotFoundError:
    print(f"Error file not found: {srt_file_path}")

def load_data(video_path, srt_file_path, dry_run=False):
  
  script_dir = pathlib.Path(__file__).parent.parent

  video_path    = os.path.join(script_dir, video_path)
  srt_file_path = os.path.join(script_dir, srt_file_path)
  output_folder = os.path.join(script_dir, 'data', 'processed') 

  if not dry_run:
    extract_frames(video_path, output_folder)
  data = extract_telemetry(srt_file_path)

  return data
