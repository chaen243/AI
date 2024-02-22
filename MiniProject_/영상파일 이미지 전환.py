import cv2
import os

# 영상 파일이 있는 디렉토리
video_directory = 'D:\\_data\\aaa\\'

# 이미지를 저장할 디렉토리 설정
output_directory = 'D:\\_data\\aaa\\'
os.makedirs(output_directory, exist_ok=True)

# 영상 파일 리스트 가져오기
video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]

for video_file in video_files:
    # VideoCapture 객체 생성
    video_path = os.path.join(video_directory, video_file)
    cap = cv2.VideoCapture(video_path)

    # 프레임 단위로 영상을 읽어서 이미지로 저장
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 이미지 저장
        image_path = os.path.join(output_directory, f'{video_file}_frame_{frame_count}.jpg')
        cv2.imwrite(image_path, frame)

        frame_count += 1

    # VideoCapture 객체 해제
    cap.release()