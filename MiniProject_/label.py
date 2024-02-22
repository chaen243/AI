from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import cv2

# Step 1: 프레임 추출
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

# Step 2: 프레임 전처리
def preprocess_frames(frames, feature_extractor):
    preprocessed_frames = []

    for frame in frames:
        inputs = feature_extractor(images=frame, return_tensors="pt")
        preprocessed_frames.append(inputs)

    return preprocessed_frames

# Step 3: ViT 모델 로딩
model_name = "google/vit-base-patch16-224-in21k"
vit_model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Step 4: 프레임을 모델에 전달
def process_frames(frames, model):
    outputs = []

    for frame in frames:
        output = model(**frame)
        outputs.append(output.logits) 

    return outputs

# Step 5: 프레임 출력을 결합
def combine_outputs(outputs):
    # 여기에서 각 프레임의 출력을 어떻게 결합할지 정의
    # 여기서는 간단하게 평균을 사용함
    combined_output = torch.mean(torch.stack(outputs), dim=0)

    return combined_output

# 비디오 파일 경로
video_path = "C:\\_data\\aaa\\NIA_SL_SEN0001_REAL02_D.mp4"

# Step 1: 프레임 추출
frames = extract_frames(video_path)

# Step 2: 프레임 전처리
preprocessed_frames = preprocess_frames(frames, feature_extractor)

# Step 4: 프레임을 모델에 전달
outputs = process_frames(preprocessed_frames, vit_model)

# Step 5: 프레임 출력을 결합
combined_output = combine_outputs(outputs)

# 최종 결과 출력
print("Final Video Level Output:", combined_output)