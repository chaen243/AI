from moviepy.editor import VideoFileClip
import os

def change_resolution(input_file, output_file, new_width, new_height):
    clip = VideoFileClip(input_file)
    new_clip = clip.resize(width=new_width, height=new_height)
    new_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

def batch_change_resolution(input_folder, output_folder, new_width, new_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):  # 혹은 다른 동영상 포맷에 따라 변경
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            change_resolution(input_path, output_path, new_width, new_height)

# 사용 예시
input_folder = "C:\\수어 데이터셋\\"
output_folder = "D:\\수어 데이터셋\\"
new_width = 1280
new_height = 720

batch_change_resolution(input_folder, output_folder, new_width, new_height)