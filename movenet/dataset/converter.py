import cv2
import numpy as np
import os

def split_video_into_frames(video_path, output_folder, num_frames=20):
    # 비디오를 로드하고 총 프레임 수를 구함
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    
    # 균일한 간격으로 프레임을 선택하기 위한 간격 계산
    interval = total_frames // num_frames
    
    # 출력 폴더 생성 (존재하지 않을 경우)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 선택된 프레임을 이미지 파일로 저장
    for i in range(num_frames):
        # 비디오를 올바른 위치로 이동
        video.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        
        # 프레임 읽기
        success, frame = video.read()
        if not success:
            break  # 읽기 실패시 중단
        
        # 이미지 파일로 저장
        cv2.imwrite(f"{output_folder}/frame_{i+1:03d}.jpg", frame)
    
    # 자원 해제
    video.release()
    print(f"Saved {num_frames} frames to {output_folder}")

# 사용 예
video_path = 'bowl_data_2.mp4'  # 비디오 파일 경로
output_folder = 'output_frames\\2'  # 출력될 폴더
split_video_into_frames(video_path, output_folder)