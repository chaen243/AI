from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
import os

def change_resolution(input_file, output_file, new_width, new_height):
    # VideoFileClip 생성 시 audio=False를 설정하여 오디오 스트림을 불러오지 않음
    clip = VideoFileClip(input_file, audio=False)
    new_clip = clip.resize(width=new_width, height=new_height)
    # write_videofile 메서드 호출 시 audio=False를 설정하여 오디오를 포함시키지 않음
    new_clip.write_videofile(output_file, audio=False)#codec="libx264", audio=False)

def process_file(input_folder, output_folder, new_width, new_height, filename):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    change_resolution(input_path, output_path, new_width, new_height)

def batch_change_resolution(input_folder, output_folder, new_width, new_height, max_workers=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".mp4") # 대소문자 구분 없이 검사
                executor.submit(process_file, input_folder, output_folder, new_width, new_height, filename)

# 사용 예시
input_folder = "C:\\hand_sign\\"
output_folder = "D:\\hand_sign\\"
new_width = 320
new_height = 320
max_workers = 23  # 원하는 스레드 수 지정

batch_change_resolution(input_folder, output_folder, new_width, new_height, max_workers=max_workers)