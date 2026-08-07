import os
os.environ['MPLBACKEND'] = 'Agg'

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="AI 정밀 골프 스윙 분석 시스템", layout="wide")

st.title("⛳ AI 정밀 골프 스윙 분석 시스템 (P1 ~ P13 자동 추출)")
st.write("스윙 영상을 업로드하시면 AI가 자동으로 주요 페이즈를 분석하고 스틸컷과 데이터를 제공합니다.")

@st.cache_resource
def load_models():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
    yolo_model = YOLO('yolov8n.pt')
    return mp_pose, pose, yolo_model

mp_pose, pose, yolo_model = load_models()

phase_info = [
    ("P1", "Address", "스윙 시작 전 정지 상태"),
    ("P2", "Start Sweep", "샤프트가 지면에 45도"),
    ("P3", "Back Alignment", "샤프트가 지면에 평행"),
    ("P4", "Start Shoulder Back", "왼팔이 지면에 평행"),
    ("P5", "Backswing Top", "헤드의 정지 (정지된 시간 측정)"),
    ("P6", "Transition", "샤프트가 지면에 135도"),
    ("P7", "DB Alignment", "샤프트가 지면에 평행"),
    ("P8", "Impact", "볼을 타격하는 지점"),
    ("P9", "Lowest Club Head", "샤프트가 지면에 45도"),
    ("P10", "DF Alignment", "샤프트가 지면에 평행"),
    ("P11", "Start Shoulder Forward", "오른팔이 지면에 평행"),
    ("P12", "Downswing Top", "최고점의 그립"),
    ("P13", "Finish", "스윙이 끝날 때의 정지 상태")
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360.0 - angle

def calculate_tilt(left, right):
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return np.arctan2(dy, dx) * 180.0 / np.pi

def get_club_angle(wrist_pos, club_head_pos):
    if not wrist_pos or not club_head_pos:
        return None
    dx = club_head_pos[0] - wrist_pos[0]
    dy = club_head_pos[1] - wrist_pos[1]
    angle = np.arctan2(-dy, dx) * 180.0 / np.pi 
    return angle % 180

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("정밀 분석 시작", type="primary"):
        with st.spinner("AI가 영상을 정밀 분석 중입니다. 잠시만 기다려주세요..."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_data = []
            frames_cache = []
            frame_idx = 0
            prev_wrist_pos = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_cache.append(image_rgb)
                results = pose.process(image_rgb)
                yolo_results = yolo_model(image_rgb, verbose=False)[0]
                
                club_head_pos = None
                for box in yolo_results.boxes:
                    if int(box.cls[0]) == 32: 
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        club_head_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
                        break
                        
                metrics = {
                    "frame_idx": frame_idx, "wrist_y": 0, "wrist_pos": None,
                    "l_shoulder_y": 0, "r_shoulder_y": 0, "shoulder_tilt": 0, 
                    "hip_tilt": 0, "lt_elbow_angle": 0, "wrist_speed": 0,
                    "club_angle": 0, "has_landmarks": False
                }
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = frame.shape
                    
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
                    
                    metrics.update({
                        "has_landmarks": True, "wrist_pos": l_wrist,
                        "wrist_y": l_wrist[1], "l_shoulder_y": l_shoulder[1],
                        "r_shoulder_y": r_shoulder[1],
                        "shoulder_tilt": round(calculate_tilt(l_shoulder, r_shoulder), 2),
                        "hip_tilt": round(calculate_tilt(l_hip, r_hip), 2),
                        "lt_elbow_angle": round(calculate_angle(l_shoulder, l_elbow, l_wrist), 2)
                    })
                    
                    if prev_wrist_pos:
                        metrics["wrist_speed"] = round(math.dist(l_wrist, prev_wrist_pos) * fps, 2)
                    
                    c_angle = get_club_angle(l_wrist, club_head_pos)
                    metrics["club_angle"] = round(c_angle, 2) if c_angle else 0
                    prev_wrist_pos = l_wrist
                    
                frame_data.append(metrics)
                frame_idx += 1
                
            cap.release()
            
            df_temp = pd.DataFrame(frame_data)
            valid = df_temp[df_temp['has_landmarks'] == True]
            
            if not valid.empty:
                p1_idx = valid.index[0] 
                p8_idx = valid['wrist_y'].idxmax() 
                backswing_data = valid.loc[p1_idx:p8_idx]
                p5_idx = backswing_data['wrist_y'].idxmin() if not backswing_data.empty else p1_idx
                follow_data = valid.loc[p8_idx:]
                p12_idx = follow_data['wrist_y'].idxmin() if not follow_data.empty else p8_idx
                p13_idx = valid.index[-1] 
                
                def find_closest_angle(data_subset, target_angle, col='club_angle'):
                    if data_subset.empty: return p1_idx
                    return (data_subset[col] - target_angle).abs().idxmin()
                    
                p2_idx = find_closest_angle(valid.loc[p1_idx:p5_idx], 45)
                p3_idx = find_closest_angle(valid.loc[p2_idx:p5_idx], 0)
                p4_subset = valid.loc[p3_idx:p5_idx]
                p4_idx = (p4_subset['wrist_y'] - p4_subset['l_shoulder_y']).abs().idxmin() if not p4_subset.empty else p3_idx
                
                p6_idx = find_closest_angle(valid.loc[p5_idx:p8_idx], 135)
                p7_idx = find_closest_angle(valid.loc[p6_idx:p8_idx], 0)
                
                p9_idx = find_closest_angle(valid.loc[p8_idx:p12_idx], 45)
                p10_idx = find_closest_angle(valid.loc[p9_idx:p12_idx], 0)
                p11_subset = valid.loc[p10_idx:p12_idx]
                p11_idx = (p11_subset['wrist_y'] - p11_subset['r_shoulder_y']).abs().idxmin() if not p11_subset.empty else p10_idx
                
                phase_indices = [p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx, p7_idx, p8_idx, p9_idx, p10_idx, p11_idx, p12_idx, p13_idx]
                
                st.subheader("📸 P1 ~ P13 단계별 스틸컷")
                cols = st.columns(4)
                for i, p_idx in enumerate(phase_indices):
                    data = frame_data[p_idx]
                    img = frames_cache[p_idx].copy()
                    results = pose.process(img)
                    if results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    with cols[i % 4]:
                        st.image(img, caption=f"{phase_info[i][0]}: {phase_info[i][1]}", use_container_width=True)
                
                st.subheader("📊 스윙 분석 결과 테이블")
                table_data = []
                for i, p_idx in enumerate(phase_indices):
                    data = frame_data[p_idx]
                    time_stamp = round(data["frame_idx"] / fps, 2) if fps > 0 else 0
                    table_data.append({
                        "Phase": phase_info[i][0], "Name": phase_info[i][1], "기준": phase_info[i][2],
                        "TimeStamp(s)": time_stamp, "Frame #": data["frame_idx"],
                        "Shoulder Tilt": data["shoulder_tilt"], "HipTilt": data["hip_tilt"],
                        "LtElbow": data["lt_elbow_angle"],
                        "ClubAngle": data["club_angle"] if data["club_angle"] != 0 else "N/A"
                    })
                df_result = pd.DataFrame(table_data)
                st.dataframe(df_result, use_container_width=True)
            else:
                st.error("영상에서 포즈를 감지하지 못했습니다. 다른 영상을 업로드해 주세요.")
