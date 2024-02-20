import os
import random
import shutil

def limit_files_in_folders(root_folder_path, limit_per_folder):
    # 루트 폴더 내의 모든 폴더를 순회합니다.
    for folder_name in os.listdir(root_folder_path):
        folder_path = os.path.join(root_folder_path, folder_name)
        
        # 폴더인 경우에만 작업을 수행합니다.
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            if len(files) > limit_per_folder:
                # 파일 목록을 무작위로 섞습니다.
                random.shuffle(files)

                files_to_keep = files[:limit_per_folder]  # limit_per_folder 개수만큼 파일을 유지합니다.
                files_to_remove = files[limit_per_folder:]  # 제거할 파일을 선택합니다.

                for file_name in files_to_remove:
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # 파일을 삭제합니다.
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 폴더를 삭제합니다.

# 루트 폴더 경로와 각 폴더당 제한할 파일의 개수를 지정합니다.
root_folder_path = "C:\\classified_video_320\\train"
limit_per_folder = 3

# 함수를 호출하여 각 폴더 내의 파일을 제한 개수로 유지합니다.
limit_files_in_folders(root_folder_path, limit_per_folder)