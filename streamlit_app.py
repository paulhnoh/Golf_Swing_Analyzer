"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인]
1. 물리적 프레임 전수 추출 (Frame-by-Frame Physical Storage):
   - 영상 전체 프레임을 temp 디렉토리에 개별 JPG 이미지로 완벽히 분리 저장 후 분석 수행.
2. 풀 프레임 뷰 보장 (Full-Frame View):
   - 임의의 크롭을 전면 폐지하고 원본 전체 화면을 출력하여 클럽 헤드, 샤프트, 공이 100% 노출되도록 함.
3. Phase 정의 맞춤형 오버레이 시각화 (Definition-Based Visual Overlay):
   - 샤프트 수직(P1), 수평(P3, P7, P10), 팔 수평(P4, P11), 각도(P2, P6, P9) 등 
     Phase 정의에 부합하는 가이드라인과 실측 각도를 직관적으로 오버레이함.
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

st.set_page_config(page_title="P1-P13 Pro Swing Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 정밀 분석 및 맞춤형 오버레이 시스템")
st.markdown("전체 화면 뷰를 통해 클럽과 공을 온전히 보장하며, 페이즈 정의에 맞춘 정밀 오버레이를 제공합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

phases_info = [
    {"phase": "P1", "name": "Address", "desc": "샤프트가 지면과 수직", "target_angle": 90, "type": "shaft_vert"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트가 지면과 45°", "target_angle": 45, "type": "shaft_deg"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트가 지면에 평행", "target_angle": 0, "type": "shaft_horiz"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔이 지면에 평행", "target_angle": 0, "type": "arm_horiz_left"},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (체공시간 측정)", "target_angle": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "desc": "샤프트가 지면과 135°", "target_angle": 135, "type": "shaft_deg"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트가 지면에 평행", "target_angle": 0, "type": "shaft_horiz"},
    {"phase": "P8", "name": "Impact", "desc": "볼을 타격하는 지점", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트가 지면과 315°", "target_angle": 315, "type": "shaft_deg"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트가 지면에 평행", "target_angle": 0, "type": "shaft_horiz"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔이 지면에 평행", "target_angle": 0, "type": "arm_horiz_right"},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (체공시간 측정)", "target_angle": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "desc": "스윙 종료 정지 상태", "target_angle": None, "type": "finish"},
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

    if 'auto_frames' not in st.session_state:
        with st.spinner("1단계: 전체 프레임을 물리적 이미지로 분리 및 전수 스캔 중..."):
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
                        # 지면 기준 각도 계산 (수평 기준 0도 ~ 90도 수직)
                        raw_angle = math.degrees(math.atan2(target[1]-wrist_pt[1], target[0]-wrist_pt[0]))
                        sa = abs(raw_angle)
                
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
            
            # 0번 프레임 쏠림 방지를 위한 구간 안전 분할 보정
            auto_f["P2"] = max(p1_idx + 2, find_closest_frame(shaft_angles, 45, auto_f["P1"], auto_f["P5"]))
            auto_f["P3"] = max(auto_f["P2"] + 2, find_closest_frame(shaft_angles, 0, auto_f["P2"], auto_f["P5"]))
            auto_f["P4"] = max(auto_f["P3"] + 2, find_closest_frame(left_arm_angles, 0, auto_f["P3"], auto_f["P5"]))
            
            auto_f["P6"] = find_closest_frame(shaft_angles, 135, auto_f["P5"], auto_f["P8"])
            auto_f["P7"] = find_closest_frame(shaft_angles, 0, auto_f["P6"], auto_f["P8"])
            auto_f["P9"] = find_closest_frame(shaft_angles, 45, auto_f["P8"], auto_f["P12"])
            auto_f["P10"] = find_closest_frame(shaft_angles, 0, auto_f["P9"], auto_f["P12"])
            auto_f["P11"] = find_closest_frame(right_arm_angles, 0, auto_f["P10"], auto_f["P12"])

            st.session_state.p5_time = calculate_peak_duration(y_left[:p8_idx], fps)
            st.session_state.p12_time = calculate_peak_duration(y_right[p8_idx:], fps)
            st.session_state.shaft_angles = shaft_angles
            st.session_state.auto_frames = auto_f
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 페이즈별 풀 프레임 정밀 오버레이 검증 뷰")
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

                    # 💡 정의 맞춤형 오버레이 시각화
                    if wrist_pt and target_pt:
                        cv2.circle(img, wrist_pt, 8, (0, 255, 255), -1)
                        cv2.circle(img, target_pt, 8, (0, 0, 255), -1)
                        
                        # Phase 정의에 따른 의미 있는 오버레이 라인 및 텍스트
                        if p['type'] in ['shaft_vert', 'shaft_horiz', 'shaft_deg']:
                            cv2.line(img, wrist_pt, target_pt, (0, 255, 0), 4)
                            dx = target_pt[0] - wrist_pt[0]
                            dy = target_pt[1] - wrist_pt[1]
                            measured_val = round(abs(math.degrees(math.atan2(dy, dx))), 1)
                            cv2.putText(img, f"Shaft Angle: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif 'arm_horiz' in p['type'] and p_res.keypoints is not None:
                            kpts = p_res.keypoints.xy[0].cpu().numpy()
                            shoulder_idx = 5 if 'left' in p['type'] else 6
                            wrist_idx = 9 if 'left' in p['type'] else 10
                            if kpts[shoulder_idx][0] > 0 and kpts[wrist_idx][0] > 0:
                                s_pt = (int(kpts[shoulder_idx][0]), int(kpts[shoulder_idx][1]))
                                w_pt = (int(kpts[wrist_idx][0]), int(kpts[wrist_idx][1]))
                                cv2.line(img, s_pt, w_pt, (0, 255, 0), 4)
                                dy = w_pt[1] - s_pt[1]
                                dx = w_pt[0] - s_pt[0]
                                measured_val = round(abs(math.degrees(math.atan2(dy, dx))), 1)
                                cv2.putText(img, f"Arm Tilt: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 💡 [핵심] 크롭을 완전히 제거하고 원본 전체 화면(Full-Frame) 유지 -> 클럽과 볼 100% 보장
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{phase_id}] {p['name']}")
                
                head_still = 0.0
                if phase_id == "P5": head_still = st.session_state.p5_time
                elif phase_id == "P12": head_still = st.session_state.p12_time

                analysis_data.append({
                    "Phase": phase_id,
                    "Name": p['name'],
                    "정의 기준 (Target)": p['desc'],
                    "목표 값": str(p['target_angle']) if p['target_angle'] is not None else "특정 변곡점",
                    "AI 측정 값": measured_val,
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
