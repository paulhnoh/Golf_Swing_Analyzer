"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인]
1. 물리적 프레임 전수 추출 (Frame-by-Frame Physical Storage):
   - 영상의 전체 프레임을 OpenCV로 열어 temp 디렉토리에 개별 JPG 이미지 파일로 
     완벽하게 분리 저장한 뒤, 저장된 이미지를 불러와서 분석을 수행함. (메모리 꼬임 방지)
2. 물리적 특징점 기반 매핑 (Heuristic Anchoring):
   - 13등분 방식 금지. 스윙의 물리적 변곡점을 기준으로 싯점 자동 매핑.
   - P8 (Impact): 손목 Y좌표가 가장 낮은 프레임.
   - P5 (Top): P1~P8 사이 왼손 Y좌표가 가장 높은 프레임.
   - P3/P7/P10 (수평): 샤프트 각도 90도/270도와 가장 가까운 프레임 검색.
3. 풀 바디 및 클럽 뷰 (Full Body & Club Cropping):
   - 클럽 헤드, 샤프트, 공이 잘리지 않도록 광폭 뷰(400px 이상 여백)로 크롭 유지.
4. 오버레이 시각화 및 전문가 미세조정 (Visual Overlay & Expert UI):
   - 샤프트 선(초록색)과 측정 각도 수치를 이미지에 직접 오버레이하고, Slider로 즉시 보정 가능하게 함.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import os
import tempfile
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="P1-P13 Pro Swing Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 정밀 분석 및 오버레이 시스템")
st.markdown("물리적 프레임 저장 방식 기반으로 클럽과 샤프트를 온전히 포착하여 오버레이 분석을 수행합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

phases_info = [
    {"phase": "P1", "name": "Address", "desc": "스윙 시작 전 정지 상태", "target_angle": 0},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트가 지면과 45°", "target_angle": 45},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트가 지면에 평행", "target_angle": 90},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔이 지면에 평행", "target_angle": None},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (체공시간 측정)", "target_angle": None},
    {"phase": "P6", "name": "Transition", "desc": "샤프트가 지면과 45°", "target_angle": 135},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트가 지면에 평행", "target_angle": 90},
    {"phase": "P8", "name": "Impact", "desc": "볼을 타격하는 지점", "target_angle": None},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트가 지면과 45°", "target_angle": 315},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트가 지면에 평행", "target_angle": 270},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔이 지면에 평행", "target_angle": None},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (체공시간 측정)", "target_angle": None},
    {"phase": "P13", "name": "Finish", "desc": "스윙이 끝날 때의 정지 상태", "target_angle": None},
]

def calculate_peak_duration(y_coords, fps=30, threshold=10.0):
    valid_y = [y for y in y_coords if not np.isnan(y)]
    if not valid_y: return 0.0
    peak_y = min(valid_y) 
    return round(len([y for y in valid_y if abs(y - peak_y) <= threshold]) / fps, 3)

def find_closest_frame(arr, target, start_idx, end_idx):
    if start_idx >= end_idx or start_idx >= len(arr): return start_idx
    sub_arr = arr[start_idx:end_idx]
    valid_indices = np.where(~np.isnan(sub_arr))[0]
    return start_idx + valid_indices[np.argmin(np.abs(np.array(sub_arr)[valid_indices] - target))] if len(valid_indices) > 0 else start_idx

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    # 원칙 1: 영상 전체를 물리적 이미지(JPG)로 개별 분리 저장 후 분석
    if 'auto_frames' not in st.session_state:
        with st.spinner("1단계: 전체 프레임을 개별 이미지 파일로 물리적 추출 및 저장 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            
            y_left, y_right, left_arm_angles, right_arm_angles, shaft_angles = [], [], [], [], []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or total_frames > 600: break
                
                # 물리적 이미지 저장
                img_path = os.path.join(frame_dir, f"frame_{total_frames:04d}.jpg")
                cv2.imwrite(img_path, frame)
                
                # 저장된 이미지를 즉시 불러와서 분석 (원칙 1 준수)
                analyzed_frame = cv2.imread(img_path)
                p_res = pose_model(analyzed_frame, verbose=False)[0]
                c_res = custom_model(analyzed_frame, verbose=False)[0]
                
                ly, ry, la, ra, sa = np.nan, np.nan, np.nan, np.nan, np.nan
                wrist_pt, is_p = None, False
                
                if p_res.keypoints is not None and len(p_res.keypoints.xy[0]) > 10:
                    kpts = p_res.keypoints.xy[0].cpu().numpy()
                    if kpts[5][0] > 0 or kpts[6][0] > 0:
                        is_p = True
                        if kpts[9][0] > 0: ly = kpts[9][1]
                        if kpts[10][0] > 0: ry = kpts[10][1]
                        if kpts[9][0] > 0 and kpts[10][0] > 0:
                            wrist_pt = ((kpts[9][0]+kpts[10][0])/2, (kpts[9][1]+kpts[10][1])/2)
                
                if is_p:
                    head, shaft = None, None
                    for box in c_res.boxes:
                        name = c_res.names[int(box.cls[0])]
                        cent = ((box.xyxy[0][0]+box.xyxy[0][2])/2, (box.xyxy[0][1]+box.xyxy[0][3])/2)
                        if name == 'head': head = cent
                        elif name == 'shaft': shaft = cent
                    target = head if head else shaft
                    if wrist_pt and target:
                        sa = abs(math.degrees(math.atan2(target[1]-wrist_pt[1], target[0]-wrist_pt[0])))
                
                y_left.append(ly); y_right.append(ry)
                left_arm_angles.append(la); right_arm_angles.append(ra); shaft_angles.append(sa)
                total_frames += 1
            cap.release()
            
            # 원칙 2: 물리적 변곡점 기반 매핑 (Heuristic Anchoring)
            valid_ly = [(i, y) for i, y in enumerate(y_left) if not np.isnan(y)]
            p8_idx = max(valid_ly, key=lambda x: x[1])[0] if valid_ly else total_frames // 2
            p1_idx = valid_ly[0][0] if valid_ly else 0
            p13_idx = total_frames - 1
            
            sub_ly = [(i, y) for i, y in enumerate(y_left[:p8_idx]) if not np.isnan(y)]
            p5_idx = min(sub_ly, key=lambda x: x[1])[0] if sub_ly else p8_idx // 2
            
            sub_ry = [(i, y) for i, y in enumerate(y_right[p8_idx:]) if not np.isnan(y)]
            p12_idx = p8_idx + min(sub_ry, key=lambda x: x[1])[0] if sub_ry else total_frames - 2

            auto_f = {"P1": p1_idx, "P5": p5_idx, "P8": p8_idx, "P12": p12_idx, "P13": p13_idx}
            auto_f["P2"] = find_closest_frame(shaft_angles, 45, auto_f["P1"], auto_f["P5"])
            auto_f["P3"] = find_closest_frame(shaft_angles, 90, auto_f["P2"], auto_f["P5"])
            auto_f["P4"] = find_closest_frame(left_arm_angles, 0, auto_f["P3"], auto_f["P5"])
            auto_f["P6"] = find_closest_frame(shaft_angles, 45, auto_f["P5"], auto_f["P8"])
            auto_f["P7"] = find_closest_frame(shaft_angles, 90, auto_f["P6"], auto_f["P8"])
            auto_f["P9"] = find_closest_frame(shaft_angles, 45, auto_f["P8"], auto_f["P12"])
            auto_f["P10"] = find_closest_frame(shaft_angles, 90, auto_f["P9"], auto_f["P12"])
            auto_f["P11"] = find_closest_frame(right_arm_angles, 0, auto_f["P10"], auto_f["P12"])

            st.session_state.p5_time = calculate_peak_duration(y_left[:p8_idx], fps)
            st.session_state.p12_time = calculate_peak_duration(y_right[p8_idx:], fps)
            st.session_state.shaft_angles = shaft_angles
            st.session_state.auto_frames = auto_f
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps
            st.session_state.scan_done = True

    # 원칙 4: 저장된 개별 이미지 불러오기 및 오버레이 시각화 (Expert UI)
    if 'scan_done' in st.session_state:
        st.subheader("📸 페이즈별 정밀 오버레이 검증 뷰 (물리적 저장 이미지 기반)")
        cols = st.columns(4)
        analysis_data = []

        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                phase_id = p['phase']
                auto_fn = st.session_state.auto_frames.get(phase_id, 0)
                fn = st.slider(f"[{phase_id}] 조정", 0, st.session_state.total_frames-1, auto_fn, key=f"slider_{phase_id}")
                
                # 물리적으로 저장된 개별 이미지 파일을 정확히 불러옴
                img_path = os.path.join(st.session_state.frame_dir, f"frame_{fn:04d}.jpg")
                img = cv2.imread(img_path)
                
                measured_angle = 0.0
                if img is not None:
                    p_res = pose_model(img, verbose=False)[0]
                    c_res = custom_model(img, verbose=False)[0]
                    
                    wrist_pt, target_pt = None, None
                    if p_res.keypoints is not None and len(p_res.keypoints.xy[0]) > 10:
                        kpts = p_res.keypoints.xy[0].cpu().numpy()
                        if kpts[9][0] > 0 and kpts[10][0] > 0:
                            wrist_pt = (int((kpts[9][0]+kpts[10][0])/2), int((kpts[9][1]+kpts[10][1])/2))
                    
                    head, shaft = None, None
                    for box in c_res.boxes:
                        name = c_res.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if name == 'head': head = cent
                        elif name == 'shaft': shaft = cent
                    target_pt = head if head else shaft

                    # 오버레이 시각화 (초록색 샤프트 선 및 각도)
                    if wrist_pt and target_pt:
                        cv2.circle(img, wrist_pt, 8, (0, 255, 255), -1)
                        cv2.circle(img, target_pt, 8, (0, 0, 255), -1)
                        cv2.line(img, wrist_pt, target_pt, (0, 255, 0), 4)
                        
                        dx = target_pt[0] - wrist_pt[0]
                        dy = target_pt[1] - wrist_pt[1]
                        measured_angle = round(abs(math.degrees(math.atan2(dy, dx))), 1)
                        cv2.putText(img, f"Angle: {measured_angle}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 원칙 3: 클럽과 공이 잘리지 않도록 광폭 뷰 크롭 (400px 여유 확보)
                    if p_res.keypoints is not None and len(p_res.keypoints.xy[0]) > 5:
                        k = p_res.keypoints.xy[0].cpu().numpy()
                        if k[5][0] > 0 and k[6][0] > 0:
                            cx, cy = int((k[5][0]+k[6][0])/2), int((k[5][1]+k[6][1])/2)
                            img = img[max(0, cy-450):min(img.shape[0], cy+450), max(0, cx-450):min(img.shape[1], cx+450)]
                    
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{phase_id}] {p['name']}")
                
                head_still = 0.0
                if phase_id == "P5": head_still = st.session_state.p5_time
                elif phase_id == "P12": head_still = st.session_state.p12_time

                analysis_data.append({
                    "Phase": phase_id,
                    "Name": p['name'],
                    "정의 기준 (Target)": p['desc'],
                    "목표 각도": p['target_angle'] if p['target_angle'] is not None else "수평/기타",
                    "AI 측정 각도": measured_angle,
                    "Frame #": fn,
                    "Time Stamp(s)": round(fn / st.session_state.fps, 2),
                    "HeadStill Time": head_still
                })

        st.divider()
        st.subheader("📊 페이즈 정의 vs AI 측정 결과 비교 검증 표")
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드", data=csv_data,
            file_name='calibrated_swing_P1_P13.csv', mime='text/csv',
        )
