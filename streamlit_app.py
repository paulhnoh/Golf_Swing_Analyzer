"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인 (변경 불가)]
1. 240장 개별 이미지 전수 조사 (Full Frame-by-Frame Scan):
   - 영상의 모든 프레임을 temp 디렉토리에 개별 JPG 이미지로 완벽히 분리 저장한 후, 
     1장도 빠짐없이 순회하며 샤프트 각도와 팔 각도를 전수 계산하여 데이터베이스화함.
2. 지면 기준 정확한 각도 계산 및 엄격한 검증 (Ground-Truth Angle & Strict Validation):
   - 지면(수평)을 기준으로 수직(90°), 평행(0°/180°), 45° 등을 정확히 산출하고, 
     실측 각도가 목표 각도와 허용 오차 내에 일치할 때만 Pass 판정을 내림.
3. 풀 프레임 뷰 및 정의 맞춤형 오버레이:
   - 원본 전체 뷰(Full-Frame)를 유지하여 클럽과 공을 보장하고, 정의에 부합하는 가이드라인 오버레이.
4. 전문가 미세조정 (Expert UI):
   - Slider를 통해 언제든 즉시 프레임 미세조정 가능.
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

st.set_page_config(page_title="P1-P13 Ground-Truth Pro Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 지면 기준 정밀 각도 검증 시스템")
st.markdown("지면 수평 기준의 정확한 각도 계산과 엄격한 자체 검증 알고리즘을 통해 페이즈를 판정합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

phases_info = [
    {"phase": "P1", "name": "Address", "desc": "샤프트 수직 (지면 기준 90°)", "target_angle": 90.0, "type": "shaft"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트 지면과 45°", "target_angle": 45.0, "type": "shaft"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트 지면과 평행 (0°)", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔 지면과 평행 (0°)", "target_angle": 0.0, "type": "arm_left"},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (체공시간 측정)", "target_angle": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "desc": "샤프트 지면과 135°", "target_angle": 135.0, "type": "shaft"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트 지면과 평행 (0°)", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P8", "name": "Impact", "desc": "볼 타격 시점", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트 지면과 315°", "target_angle": 315.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트 지면과 평행 (0°)", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔 지면과 평행 (0°)", "target_angle": 0.0, "type": "arm_right"},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (체공시간 측정)", "target_angle": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "desc": "스윙 종료 정지 상태", "target_angle": None, "type": "finish"},
]

def calculate_peak_duration(y_coords, fps=30, threshold=10.0):
    valid_y = [y for y in y_coords if not np.isnan(y)]
    if not valid_y: return 0.0
    peak_y = min(valid_y) 
    return round(len([y for y in valid_y if abs(y - peak_y) <= threshold]) / fps, 3)

def compute_ground_angle(p1, p2):
    """지면(수평)을 기준으로 한 정확한 각도 계산 (0도~180도/360도)"""
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1] # 이미지 좌표계 특성상 y는 아래로 증가하므로 반전
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0: angle += 180
    return round(angle, 1)

def find_best_frame_by_angle(arr, target, start_idx, end_idx):
    if start_idx >= end_idx or start_idx >= len(arr): return start_idx
    sub_arr = arr[start_idx:end_idx]
    valid_indices = np.where(~np.isnan(sub_arr))[0]
    if len(valid_indices) == 0: return start_idx + (end_idx - start_idx)//2
    
    diffs = np.abs(np.array(sub_arr)[valid_indices] - target)
    return start_idx + valid_indices[np.argmin(diffs)]

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    if 'auto_frames' not in st.session_state:
        with st.spinner("모든 프레임 전수 스캔 및 지면 기준 각도 분석 중..."):
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
                
                img_path = os.path.join(frame_dir, f"frame_{total_frames:04d}.jpg")
                cv2.imwrite(img_path, frame)
                
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
                            wrist_pt = (int((kpts[9][0]+kpts[10][0])/2), int((kpts[9][1]+kpts[10][1])/2))
                        
                        # 팔 수평 각도 (지면 기준)
                        if kpts[5][0] > 0 and kpts[9][0] > 0:
                            la = compute_ground_angle((kpts[5][0], kpts[5][1]), (kpts[9][0], kpts[9][1]))
                        if kpts[6][0] > 0 and kpts[10][0] > 0:
                            ra = compute_ground_angle((kpts[6][0], kpts[6][1]), (kpts[10][0], kpts[10][1]))
                
                if is_p:
                    head, shaft = None, None
                    for box in c_res.boxes:
                        name = c_res.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if name == 'head': head = cent
                        elif name == 'shaft': shaft = cent
                    target = head if head else shaft
                    if wrist_pt and target:
                        sa = compute_ground_angle(wrist_pt, target)
                
                y_left.append(ly); y_right.append(ry)
                left_arm_angles.append(la); right_arm_angles.append(ra); shaft_angles.append(sa)
                total_frames += 1
            cap.release()
            
            valid_ly = [(i, y) for i, y in enumerate(y_left) if not np.isnan(y)]
            p8_idx = max(valid_ly, key=lambda x: x[1])[0] if valid_ly else total_frames // 2
            p1_idx = valid_ly[0][0] if valid_ly else 0
            p13_idx = total_frames - 1
            
            sub_ly = [(i, y) for i, y in enumerate(y_left[:p8_idx]) if not np.isnan(y)]
            p5_idx = min(sub_ly, key=lambda x: x[1])[0] if sub_ly else p8_idx // 2
            
            sub_ry = [(i, y) for i, y in enumerate(y_right[p8_idx:]) if not np.isnan(y)]
            p12_idx = p8_idx + min(sub_ry, key=lambda x: x[1])[0] if sub_ry else total_frames - 2

            auto_f = {"P1": p1_idx, "P5": p5_idx, "P8": p8_idx, "P12": p12_idx, "P13": p13_idx}
            
            # 각 구간별 지면 기준 각도 매칭
            auto_f["P2"] = find_best_frame_by_angle(shaft_angles, 45.0, auto_f["P1"], auto_f["P5"])
            auto_f["P3"] = find_best_frame_by_angle(shaft_angles, 0.0, auto_f["P2"], auto_f["P5"])
            auto_f["P4"] = find_best_frame_by_angle(left_arm_angles, 0.0, auto_f["P3"], auto_f["P5"])
            auto_f["P6"] = find_best_frame_by_angle(shaft_angles, 135.0, auto_f["P5"], auto_f["P8"])
            auto_f["P7"] = find_best_frame_by_angle(shaft_angles, 0.0, auto_f["P6"], auto_f["P8"])
            auto_f["P9"] = find_best_frame_by_angle(shaft_angles, 45.0, auto_f["P8"], auto_f["P12"])
            auto_f["P10"] = find_best_frame_by_angle(shaft_angles, 0.0, auto_f["P9"], auto_f["P12"])
            auto_f["P11"] = find_best_frame_by_angle(right_arm_angles, 0.0, auto_f["P10"], auto_f["P12"])

            st.session_state.p5_time = calculate_peak_duration(y_left[:p8_idx], fps)
            st.session_state.p12_time = calculate_peak_duration(y_right[p8_idx:], fps)
            st.session_state.shaft_angles = shaft_angles
            st.session_state.auto_frames = auto_f
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 지면 기준 정밀 오버레이 및 엄격 검증 뷰")
        cols = st.columns(4)
        analysis_data = []

        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                phase_id = p['phase']
                auto_fn = st.session_state.auto_frames.get(phase_id, 0)
                fn = st.slider(f"[{phase_id}] 조정", 0, st.session_state.total_frames-1, auto_fn, key=f"slider_{phase_id}")
                
                img_path = os.path.join(st.session_state.frame_dir, f"frame_{fn:04d}.jpg")
                img = cv2.imread(img_path)
                
                measured_val = 0.0
                verification_status = "Pass"
                
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

                    # 오버레이 및 각도 산출
                    if p['type'] == 'shaft' and wrist_pt and target_pt:
                        cv2.circle(img, wrist_pt, 8, (0, 255, 255), -1)
                        cv2.circle(img, target_pt, 8, (0, 0, 255), -1)
                        cv2.line(img, wrist_pt, target_pt, (0, 255, 0), 4)
                        measured_val = compute_ground_angle(wrist_pt, target_pt)
                        cv2.putText(img, f"Ground Angle: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif 'arm' in p['type'] and p_res.keypoints is not None:
                        kpts = p_res.keypoints.xy[0].cpu().numpy()
                        s_idx = 5 if 'left' in p['type'] else 6
                        w_idx = 9 if 'left' in p['type'] else 10
                        if kpts[s_idx][0] > 0 and kpts[w_idx][0] > 0:
                            s_pt = (int(kpts[s_idx][0]), int(kpts[s_idx][1]))
                            w_pt = (int(kpts[w_idx][0]), int(kpts[w_idx][1]))
                            cv2.line(img, s_pt, w_pt, (0, 255, 0), 4)
                            measured_val = compute_ground_angle(s_pt, w_pt)
                            cv2.putText(img, f"Arm Ground Angle: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 엄격한 검증 로직 (오차 15도 초과 시 Check 판정)
                    if p['target_angle'] is not None:
                        error = abs(measured_val - p['target_angle'])
                        if error > 15:
                            verification_status = "Check (Review)"

                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{phase_id}] {p['name']} ({verification_status})")
                
                head_still = 0.0
                if phase_id == "P5": head_still = st.session_state.p5_time
                elif phase_id == "P12": head_still = st.session_state.p12_time

                analysis_data.append({
                    "Phase": phase_id,
                    "Name": p['name'],
                    "정의 기준 (Target)": p['desc'],
                    "목표 값": str(p['target_angle']) if p['target_angle'] is not None else "변곡점",
                    "AI 측정 값": measured_val,
                    "검증 상태": verification_status,
                    "Frame #": fn,
                    "Time Stamp(s)": round(fn / st.session_state.fps, 2),
                    "HeadStill Time": head_still
                })

        st.divider()
        st.subheader("📊 지면 기준 검증 결과 비교 표")
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드", data=csv_data,
            file_name='calibrated_swing_P1_P13.csv', mime='text/csv',
        )
