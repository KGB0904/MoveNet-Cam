import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import json
# MoveNet 모델 로드
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4").signatures['serving_default']

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, 192, 192)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.int32)
    return image

def get_keypoints(image_path):
    image = load_and_preprocess_image(image_path)
    keypoints = movenet(image)['output_0']
    keypoints = tf.squeeze(keypoints, axis=0)
    return keypoints.numpy().flatten()

base_dir = 'output_frames'  # 기본 디렉토리 설정

sequences = []
labels = []

# 가정: 라벨이 폴더 이름에 해당하며, 이미지는 frame_001.jpg에서 frame_020.jpg까지 순서대로 있음
for label in sorted(os.listdir(base_dir)):
    sequence_dir = os.path.join(base_dir, label)
    if not os.path.isdir(sequence_dir):
        continue  # 디렉토리가 아니면 건너뛰기
    
    sequence_keypoints = []
    for i in range(1, 21):  # 1부터 20까지 반복
        frame_filename = f'frame_{i:03d}.jpg'  # 001, 002, ... 형태로 포매팅
        image_path = os.path.join(sequence_dir, frame_filename)
        keypoints = get_keypoints(image_path)
        sequence_keypoints.append(keypoints)
    
    sequences.append(sequence_keypoints)
    labels.append(1)  # 폴더 이름을 라벨로 사용

sequences = np.array(sequences)
labels = np.array(labels)

sequences_list = sequences.tolist()
labels_list = labels.tolist()

# JSON 파일로 저장
data = {'sequences': sequences_list, 'labels': labels_list}
with open('sequence_data.json', 'w') as json_file:
    json.dump(data, json_file)