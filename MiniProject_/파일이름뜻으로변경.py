import pandas as pd

# CSV 파일 경로
csv = 'C:\\mini_project\\new_csv_file.csv'

# CSV 파일 읽기
df = pd.read_csv(csv)

# 파일 이름으로 사용할 컬럼 이름, 예: 'video_title'
column_name = df['한국어']

import os

# 영상 파일들이 있는 디렉토리
video_directory = 'c:\\hand_sign'
new_directory = 'D:\\hand_sign'

# 영상 파일 리스트 가져오기
video_files = os.listdir(video_directory)

# 파일 이름 변경
for i, video_file in enumerate(video_files):
    # 새 파일 이름 생성 (CSV 컬럼 값 + 원래 파일 확장자)
    new_name = df[column_name].iloc[i] + os.path.splitext(video_file)[1]
    
    # 원래 파일 경로와 새 파일 경로
    original_path = os.path.join(video_directory, video_file)
    new_path = os.path.join(new_directory, new_name)
    
    # 파일 이름 변경
    os.rename(original_path, new_path)
    print(f"Renamed '{video_file}' to '{new_name}'")
    
    