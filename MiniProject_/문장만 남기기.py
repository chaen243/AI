import pandas as pd
import os

# 라벨 파일과 영상 파일이 저장된 디렉토리 경로 설정
label_file_path = 'D:\\bbb\\KETI-2017-SL-Annotation-v2_1.csv'  # 라벨 파일 경로
video_directory_path = 'D:\\수어 데이터셋\\6001~8280(영상)'  # 영상 파일이 저장된 디렉토리 경로

# CSV 파일 로드
df = pd.read_csv(label_file_path)

# '방향'이 정면이고 '타입'이 문장인 행만 필터링
filtered_df = df[(df['방향'] == '정면') & (df['타입(단어/문장)'] == '문장')]

# 필터링된 데이터를 기반으로 영상 파일 목록 생성
used_files = set(filtered_df['파일명'])

# 전체 파일 목록에서 사용되지 않는 파일 찾기
all_files = set(os.listdir(video_directory_path))
unused_files = all_files - used_files

# 사용되지 않는 파일 삭제
for file in unused_files:
    os.remove(os.path.join(video_directory_path, file))

# 필터링된 데이터를 새로운 CSV 파일로 저장
filtered_df.to_csv('D:\\bbb\\filtered_label_file.csv', index=False)