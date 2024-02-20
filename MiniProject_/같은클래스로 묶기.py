import pandas as pd
import os
from shutil import copy2

# 필터링된 라벨 CSV 파일 로드
filtered_label_path = 'D:\\bbb\\filtered_label_file.csv'
df = pd.read_csv(filtered_label_path)

# '한국어' 컬럼을 기준으로 고유한 클래스 할당
df['클래스'] = df['한국어'].factorize()[0] + 496

# 클래스별 폴더 생성 및 영상 파일 복사
video_directory_path = 'D:\\수어 데이터셋\\6001~8280(영상)'  # 원본 영상 파일이 저장된 디렉토리
class_directory_path = 'D:\hand_sign'  # 클래스별 폴더를 저장할 디렉토리

if not os.path.exists(class_directory_path):
    os.makedirs(class_directory_path)

for index, row in df.iterrows():
    class_folder = os.path.join(class_directory_path, str(row['클래스']))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    # 영상 파일을 해당 클래스 폴더로 복사
    source_path = os.path.join(video_directory_path, row['파일명'])
    destination_path = os.path.join(class_folder, row['파일명'])
    copy2(source_path, destination_path)

# 클래스 정보를 포함한 새로운 CSV 파일 저장
df.to_csv('D:\\bbb\\filtered_label_file.csv', index=False)