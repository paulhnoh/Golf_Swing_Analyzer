"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인 (변경 불가)]
1. 240장 개별 이미지 전수 조사 (Full Frame-by-Frame Scan):
   - 영상의 모든 프레임을 temp 디렉토리에 개별 JPG 이미지로 완벽히 분리 저장한 후, 
     1장도 빠짐없이 순회하며 샤프트와 팔 각도를 전수 계산하여 데이터베이스화함.
2. 양발 기준 가상 지면선 정의 (Virtual Ground Line by Feet):
   - 왼발목(Left Ankle)과 오른발목(Right Ankle) 좌표를 연결한 선을 '가상 지면선'으로 정의하고, 
     이 지면선을 기준으로 샤프트 및 팔의 상대 각도를 정밀 산출함.
3. 엄격한 근사치 검색 및 자체 검증:
   - 페이즈 정의(수직 90°, 45°, 수평 0° 등)에 가장 근사한 프레임을 전수 데이터에서 검색함.
4. 풀 프레임 뷰 및 가상 지면선 오버레이 시각화:
   - 클럽과 공이 잘리지 않도록 풀 프레임 뷰를 유지하고, 양발을 잇는 지면선(주황색)과 
     샤프트/팔 가이드라인(초록색)을 함께 오버레이함.
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

st.set_page_config(page_title="P1-P13 Virtual Ground Pro Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 양발 기준 가상 지면선 정밀 분석 시스템")
st.markdown("양발 발목 라인을 연결한 '가상 지면선'을 기준으로 샤프트와 팔의 각도를 정밀하게 측정합니다.")

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

def compute_relative_angle(p1, p2, ground_p1, ground_p2):
    """양발을 이은 가상 지면선을 기준(0도)으로 하여 객체(샤프트/팔)의 상대 각도 계산"""
    # 가상 지면선의 벡터 및 각도
    g_dx = ground_p2[0] - ground_p1[0]
    g_dy = ground_p1[1] - ground_p2[1]
    ground_angle = math.degrees(math.atan2(g_dy, g_dx))
    
    # 측정 대상 선분의 벡터 및 각도
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    target_angle = math.degrees(math.atan2(dy, dx))
    
    # 지면선 기준 상대 각도 차이 산출
    rel_angle = target_angle - ground_angle
    # 각도를 0~180도 범위로 정규화
    while rel_angle < 0: rel_angle += 180
    while rel_angle >= 180: rel_angle -= 180
    return round(rel_angle, 1)

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
        with st.spinner("240장 전수 스캔 중: 양발 기준 가상 지면선 및 상대 각도 분석 중..."):
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
                wrist_pt, target_pt, feet_ground = None, None, None
                
                if p_res.keypoints is not None and len(p_res.keypoints.xy[0]) > 16:
                    kpts = p_res.keypoints.xy[0].cpu().numpy()
                    # 15: 왼쪽 발목(Left Ankle), 16: 오른쪽 발목(Right Ankle) (또는 발끝 랜드마크 활용 가능)
                    l_ankle = (int(kpts[15][0]), int(kpts[15][1])) if kpts[15][0] > 0 else None
                    r_ankle = (int(kpts[16][0]), int(kpts[16][1])) if kpts[16][0] > 0 else None
                    
                    if l_ankle and r_ankle:
                        feet_ground = (l_ankle, r_ankle)
                    
                    if kpts[5][0] > 0 or kpts[6][0] > 0:
                        if kpts[9][0] > 0: ly = kpts[9][1]
                        if kpts[10][0] > 0: ry = kpts[10][1]
                        if kpts[9][0] > 0 and kpts[10][0] > 0:
                            wrist_pt = (int((kpts[9][0]+kpts[10][0])/2), int((kpts[9][1]+kpts[10][1])/2))
                        
                        # 팔 각도 계산 (가상 지면선 기준)
                        if feet_ground and kpts[5][0] > 0 and kpts[9][0] > 0:
                            la = compute_relative_angle((kpts[5][0], kpts[5][1]), (kpts[9][0], kpts[9][1]), feet_ground[0], feet_ground[1])
                        if feet_ground and kpts[6][0] > 0 and kpts[10][0] > 0:
                            ra = compute_relative_angle((kpts[6][0], kpts[6][1]), (kpts[10][0], kpts[10][1]), feet_ground[0], feet_ground[1])
                
                if wrist_pt:
                    head, shaft = None, None
                    for box in c_res.boxes:
                        name = c_res.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if name == 'head': head = cent
                        elif name == 'shaft': shaft = cent
                    target_pt = head if head else shaft
                    if target_pt and feet_ground:
                        sa = compute_relative_angle(wrist_pt, target_pt, feet_ground[0], feet_ground[1])
                
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
            
            # 가상 지면선 기준 각도로 최적 프레임 검색
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
        st.subheader("📸 양발 기준 가상 지면선 오버레이 및 검증 뷰")
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
                    
                    kpts = p_res.keypoints.xy[0].cpu().numpy() if (p_res.keypoints is not None and len(p_res.keypoints.xy[0]) > 16) else None
                    l_ankle = (int(kpts[15][0]), int(kpts[15][1])) if kpts and kpts[15][0] > 0 else None
                    r_ankle = (int(kpts[16][0]), int(kpts[16][1])) if kpts and kpts[16][0] > 0 else None
                    
                    # 💡 [가상 지면선 오버레이] 양발목을 잇는 주황색 선 표시
                    if l_ankle and r_ankle:
                        cv2.line(img, l_ankle, r_ankle, (0, 140, 255), 3) # 주황색 지면선
                        cv2.putText(img, "Virtual Ground Line", (l_ankle[0], l_ankle[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                    
                    wrist_pt, target_pt = None, None
                    if kpts is not None and kpts[9][0] > 0 and kpts[10][0] > 0:
                        wrist_pt = (int((kpts[9][0]+kpts[10][0])/2), int((kpts[9][1]+kpts[10][1])/2))
                    
                    head, shaft = None, None
                    for box in c_res.boxes:
                        name = c_res.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if name == 'head': head = cent
                        elif name == 'shaft': shaft = cent
                    target_pt = head if head else shaft

                    # 각도 측정 및 오버레이
                    if p['type'] == 'shaft' and wrist_pt and target_pt:
                        cv2.circle(img, wrist_pt, 8, (0, 255, 255), -1)
                        cv2.circle(img, target_pt, 8, (0, 0, 255), -1)
                        cv2.line(img, wrist_pt, target_pt, (0, 255, 0), 4) # 초록색 샤프트 라인
                        if l_ankle and r_ankle:
                            measured_val = compute_relative_angle(wrist_pt, target_pt, l_ankle, r_ankle)
                        cv2.putText(img, f"Rel Shaft: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif 'arm' in p['type'] and kpts is not None:
                        s_idx = 5 if 'left' in p['type'] else 6
                        w_idx = 9 if 'left' in p['type'] else 10
                        if kpts[s_idx][0] > 0 and kpts[w_idx][0] > 0:
                            s_pt = (int(kpts[s_idx][0]), int(kpts[s_idx][1]))
                            w_pt = (int(kpts[w_idx][0]), int(kpts[w_idx][1]))
                            cv2.line(img, s_pt, w_pt, (0, 255, 0), 4)
                            if l_ankle and r_ankle:
                                measured_val = compute_relative_angle(s_pt, w_pt, l_ankle, r_ankle)
                            cv2.putText(img, f"Rel Arm: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 자체 검증 로직 (목표 각도와 오차 15도 초과 시 Check 판정)
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
        st.subheader("📊 양발 기준 가상 지면선 검증 결과 표")
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드", data=csv_data,
            file_name='calibrated_swing_P1_P13.csv', mime='text/csv',
        )
