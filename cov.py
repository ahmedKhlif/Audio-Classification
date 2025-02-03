import os
import subprocess
from tqdm import tqdm

input_folder = "wavfiles/"
output_folder = "fixed_wavfiles/"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all WAV files in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Convert to PCM 16-bit WAV format using ffmpeg
        command = f'ffmpeg -i "{input_path}" -acodec pcm_s16le -ar 44100 "{output_path}" -y'
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("All WAV files converted successfully!")
