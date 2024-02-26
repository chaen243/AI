import cv2
import os
import numpy as np
import tensorflow as tf


'''

def load_and_process_video_opencv(file_path, target_size=(224, 224), sample_rate=4):
    """
    비디오 파일로부터 프레임을 추출하고 전처리합니다.

    Args:
    - file_path: 처리할 비디오 파일 경로
    - target_size: 출력 프레임의 크기 (width, height)
    - sample_rate: 프레임 샘플링 비율 (예: 5는 비디오의 매 5번째 프레임을 취함)

    Returns:
    - video_frames: 전처리된 프레임의 배열
    """
    cap = cv2.VideoCapture(file_path)
    video_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 샘플링 비율에 따라 프레임 선택
        if frame_count % sample_rate == 0:
            # BGR에서 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 프레임 크기 조정
            frame = cv2.resize(frame, target_size)
            # 정규화
            frame = frame / 255.0
            video_frames.append(frame)
        
        frame_count += 1

    cap.release()
    video_frames = np.array(video_frames, dtype=np.float32)

    return video_frames

def load_videos_from_folder(folder_path, target_size=(224, 224), sample_rate=4):
    """
    지정된 폴더 내 모든 비디오를 처리합니다.

    Args:
    - folder_path: 비디오 파일들이 있는 폴더 경로
    - target_size: 출력 프레임의 크기 (width, height)
    - sample_rate: 프레임 샘플링 비율

    Returns:
    - videos: 모든 비디오의 전처리된 프레임을 포함하는 리스트
    """
    videos = []
    filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for file_path in filenames:
        video_frames = load_and_process_video_opencv(file_path, target_size, sample_rate)
        videos.append(video_frames)

    return videos

folder_path = 'D:\\aaa'  # 비디오가 있는 폴더 경로
videos = load_videos_from_folder(folder_path)

# element_spec 명시를 위한 데이터 구조 확인 및 정의
# 여기서는 videos 리스트의 구조에 기반하여 element_spec를 정의해야 합니다.
# 예를 들어, videos가 (N, H, W, C) 형태의 비디오 프레임 배열을 포함한다고 가정할 때,
# 각 비디오의 프레임 수(N)가 다를 수 있으므로, 가장 일반적인 형태를 사용해야 합니다.
# 예시: tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)

# TensorFlow 데이터셋 생성
dataset = tf.data.Dataset.from_generator(lambda: iter(videos), output_signature=tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32))

# 배치 처리
batch_size = 32
dataset = dataset.batch(batch_size)

# 데이터셋 저장 경로
save_path = 'E:\\hands_'

# 데이터셋 저장
tf.data.experimental.save(dataset, save_path, compression=None)
'''
save_path = 'E:\\hands_'


tf.data.experimental.load(save_path)