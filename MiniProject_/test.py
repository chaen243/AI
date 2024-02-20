import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
#import remotezip as rz

import tensorflow as tf

import imageio
from IPython import display
from urllib import request
#from tensorflow_docs.vis import embed

# 1. 파일 경로 수집
import cv2
import tensorflow as tf
import numpy as np
import os

folder_path = 'D:\\aaa\\'
file_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.MP4')]

# 데이터셋 생성

def load_and_preprocess_video(file_path, frame_size=(224, 224)):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
        frame = cv2.resize(frame, frame_size)  # 프레임 크기 조정
        frame = frame / 255.0  # 정규화
        frames.append(frame)
    cap.release()
    return np.array(frames)

def create_dataset_from_folder(folder_path, frame_size=(224, 224)):
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.MP4')]
    videos_data = []
    for video_file in video_files:
        video_data = load_and_preprocess_video(video_file, frame_size=frame_size)
        videos_data.append(video_file)
    return videos_data  # 이 리스트를 사용하여 데이터셋 생성

def serialize_example(feature):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=feature.flatten()))
    feature_proto = tf.train.Features(feature={'video': feature})
    example_proto = tf.train.Example(features=feature_proto)
    return example_proto.SerializeToString()

def save_as_tfrecord(data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for video_data in data:
            serialized_data = serialize_example(video_data)
            writer.write(serialized_data)
            
# 폴더 경로 지정
folder_path = 'D:\\bbb\\'

# 데이터셋 생성
videos_data = create_dataset_from_folder(folder_path)

# TFRecord 파일로 저장
save_as_tfrecord(videos_data, 'videos_dataset.tfrecord')            