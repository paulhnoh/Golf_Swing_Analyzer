"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인]
1. 전수 좌표 DB화 및 매칭 (Full Coordinate DB Mapping):
   - 240장 전체를 스캔하며 손목, 어깨, 샤프트의 실제 X/Y 좌표를 DataFrame에 모두 저장.
   - UI에서 미세조정 시 YOLO를 재구동하지 않고 DB의 좌표를 즉시 불러와 렌더링(선 사라짐 원천 차단).
2. 360도 스윙 벡터 수학적 완벽 동기화 (Math Correction):
   - 대표님 다이어그램 적용: Left(0°), Down(90°), Right(180°), Up(270°), Top-Left(315°), Bottom-Right(135°).
3. 동적 객체 트래킹 (Dynamic Object Tracking):
   - P1 클럽 길이 반경 내에서, 이전 프레임 위치와 가장 가까운 객체를 연속 추적하여 가짜 마커 회피.
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

st.set_page_config(page_title="P1-P13 Perfect DB Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 전수 좌표 DB 및 다이나믹 오버레이 시스템")
st.markdown("모든 프레임의 좌표를 DB화하여 미세조정 시 오버레이가 즉시, 그리고 절대 사라지지 않고 반영됩니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

# 대표님 다이어그램 기준 완벽 동기화 체계
phases_info = [
    {"phase": "P1", "name": "Address", "desc": "샤프트 지면 수직", "target_angle": 90.0, "type": "shaft"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트 45°", "target_angle": 45.0, "type": "shaft"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트 좌측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔 좌측 수평", "target_angle": 0.0, "type": "arm_left"},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (탑)", "target_angle": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "desc": "샤프트 다운스윙 315°", "target_angle": 315.0, "type": "shaft"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트 좌측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P8", "name": "Impact", "desc": "볼 타격 (최저점)", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트 릴리스 135°", "target_angle": 135.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트 우측 수평", "target_angle": 180.0, "type": "shaft"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔 우측 수평", "target_angle": 180.0, "type": "arm_right"},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (피니시 진입)", "target_angle": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "desc": "스윙 종료 정지", "target_angle": None, "type": "finish"},
]

def calculate_peak_duration(y_coords, fps=30, threshold=10.0):
    valid_y = [y for y in y_coords if not np.isnan(y)]
    if not valid_y: return 0.0
    peak_y = min(valid_y) 
    return round(len([y for y in valid_y if abs(y - peak_y) <= threshold]) / fps, 3)

def compute_relative_angle(p1, p2, ground_p1, ground_p2):
    """대표님의 360도 스윙 벡터 다이어그램 완벽 동기화 로직"""
    if ground_p1[0] > ground_p2[0]:
        ground_p1, ground_p2 = ground_p2, ground_p1
        
    g_dx = ground_p2[0] - ground_p1[0]
    g_dy = ground_p2[1] - ground_p1[1]
    ground_tilt = math.atan2(g_dy, g_dx)
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    cos_t = math.cos(-ground_tilt)
    sin_t = math.sin(-ground_tilt)
    
    # 💡 핵심 수정: X축을 반전(-rx)하여 Left가 0도, Right가 180도가 되도록 수학적 보정
    rx = -(dx * cos_t - dy * sin_t)
    ry = dx * sin_t + dy * cos_t
    
    angle = math.degrees(math.atan2(ry, rx))
    if angle < 0: angle += 360
    return round(angle, 1)

def angle_diff(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def find_best_frame_from_db(db_df, col_name, target_val, start_f, end_f):
    sub = db_df[(db_df['Frame'] >= start_f) & (db_df['Frame'] <= end_f)]
    if sub.empty: return start_f
    valid = sub.dropna(subset=[col_name])
    if valid.empty: return start_f
    diffs = valid[col_name].apply(lambda x: angle_diff(x, target_val))
    best_row = valid.loc[diffs.idxmin()]
    return int(best_row['Frame'])

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    if 'auto_frames' not in st.session_state:
        with st.spinner("240장 전수 좌표 DB 구축 및 다이나믹 트래커 가동 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            
            db_records = []
            p1_ground = None 
            max_allowed_dist = None
            
            # P1 초기화
            temp_cap = cv2.VideoCapture(tfile.name)
            ret, first_frame = temp_cap.read()
            if ret:
                h_img, w_img, _ = first_frame.shape
                p_res_first = pose_model(first_frame, verbose=False)[0]
                c_res_first = custom_model(first_frame, verbose=False)[0]
                
                wrist_pt_first = None
                if p_res_first.keypoints is not None and len(p_res_first.keypoints.xy) > 0:
                    kpts_f = p_res_first.keypoints.xy[0].cpu().numpy()
                    if len(kpts_f) > 16 and kpts_f[15][0] > 0 and kpts_f[16][0] > 0:
                        p1_ground = ((int(kpts_f[15][0]), int(kpts_f[15][1])), (int(kpts_f[16][0]), int(kpts_f[16][1])))
                    if len(kpts_f) > 10 and kpts_f[9][0] > 0 and kpts_f[10][0] > 0:
                        wrist_pt_first = (int((kpts_f[9][0]+kpts_f[10][0])/2), int((kpts_f[9][1]+kpts_f[10][1])/2))
                
                if not p1_ground:
                    p1_ground = ((int(w_img * 0.35), int(h_img * 0.85)), (int(w_img * 0.65), int(h_img * 0.85)))
                
                ref_length = None
                if wrist_pt_first:
                    for box in c_res_first.boxes:
                        name = c_res_first.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if name in ['head', 'shaft']:
                            ref_length = math.hypot(cent[0] - wrist_pt_first[0], cent[1] - wrist_pt_first[1])
                            break
                if not ref_length: ref_length = w_img * 0.3
                max_allowed_dist = ref_length * 1.3
            temp_cap.release()
            st.session_state.fixed_ground = p1_ground

            prev_target_pt = None # 동적 트래커용 변수

            # 240 프레임 전수 스캔 및 DB 좌표 기록
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or total_frames > 600: break
                
                img_path = os.path.join(frame_dir, f"frame_{total_frames:04d}.jpg")
                cv2.imwrite(img_path, frame)
                
                analyzed_frame = cv2.imread(img_path)
                p_res = pose_model(analyzed_frame, verbose=False)[0]
                c_res = custom_model(analyzed_frame, verbose=False)[0]
                
                ly, ry, la, ra, sa = np.nan, np.nan, np.nan, np.nan, np.nan
                wx, wy, tx, ty, lsx, lsy, rsx, rsy = [np.nan]*8
                
                if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                    kpts = p_res.keypoints.xy[0].cpu().numpy()
                    conf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints.conf is not None else np.ones(len(kpts))
                    
                    if len(kpts) > 10:
                        if kpts[9][0] > 0 and conf[9] > 0.4: ly = kpts[9][1]
                        if kpts[10][0] > 0 and conf[10] > 0.4: ry = kpts[10][1]
                        
                        if kpts[5][0] > 0 and kpts[9][0] > 0:
                            lsx, lsy = int(kpts[5][0]), int(kpts[5][1])
                            lwx, lwy = int(kpts[9][0]), int(kpts[9][1])
                            la = compute_relative_angle((lsx, lsy), (lwx, lwy), p1_ground[0], p1_ground[1])
                        
                        if kpts[6][0] > 0 and kpts[10][0] > 0:
                            rsx, rsy = int(kpts[6][0]), int(kpts[6][1])
                            rwx, rwy = int(kpts[10][0]), int(kpts[10][1])
                            ra = compute_relative_angle((rsx, rsy), (rwx, rwy), p1_ground[0], p1_ground[1])
                        
                        pts = []
                        if kpts[9][0] > 0 and conf[9] > 0.4: pts.append(kpts[9])
                        if kpts[10][0] > 0 and conf[10] > 0.4: pts.append(kpts[10])
                        
                        if pts:
                            wrist_pt = (int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts])))
                            wx, wy = wrist_pt[0], wrist_pt[1]
                            valid_targets = []
                            
                            for box in c_res.boxes:
                                c = float(box.conf[0])
                                if c < 0.1: continue 
                                cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                                dist = math.hypot(cent[0] - wrist_pt[0], cent[1] - wrist_pt[1])
                                
                                if dist < max_allowed_dist:
                                    valid_targets.append(cent)
                            
                            if valid_targets:
                                # 💡 [동적 트래커] 이전 프레임 위치에서 가장 가까운 객체 추적
                                if prev_target_pt is not None:
                                    target_pt = min(valid_targets, key=lambda p: math.hypot(p[0]-prev_target_pt[0], p[1]-prev_target_pt[1]))
                                else:
                                    target_pt = max(valid_targets, key=lambda p: math.hypot(p[0]-wrist_pt[0], p[1]-wrist_pt[1]))
                                
                                prev_target_pt = target_pt
                                tx, ty = target_pt[0], target_pt[1]
                                sa = compute_relative_angle(wrist_pt, target_pt, p1_ground[0], p1_ground[1])
                            else:
                                prev_target_pt = None
                
                # DB에 모든 시각화 좌표 100% 저장 (오버레이 소실 방지)
                db_records.append({
                    "Frame": total_frames,
                    "LeftHandY": ly, "RightHandY": ry,
                    "ShaftAngle": sa, "LtArmAngle": la, "RtArmAngle": ra,
                    "WristX": wx, "WristY": wy, "TargetX": tx, "TargetY": ty,
                    "LShoulderX": lsx, "LShoulderY": lsy, "RShoulderX": rsx, "RShoulderY": rsy
                })
                total_frames += 1
            cap.release()
            
            df_db = pd.DataFrame(db_records)
            df_db['LeftHandY_Smooth'] = df_db['LeftHandY'].rolling(window=5, min_periods=1, center=True).median()
            df_db['RightHandY_Smooth'] = df_db['RightHandY'].rolling(window=5, min_periods=1, center=True).median()
            st.session_state.df_db = df_db
            
            valid_ly = df_db.dropna(subset=['LeftHandY_Smooth'])
            p1_idx = int(valid_ly.iloc[0]['Frame']) if not valid_ly.empty else 0
            
            sub_p5 = valid_ly[(valid_ly['Frame'] >= p1_idx) & (valid_ly['Frame'] <= total_frames * 0.6)]
            p5_idx = int(sub_p5.loc[sub_p5['LeftHandY_Smooth'].idxmin()]['Frame']) if not sub_p5.empty else total_frames // 4
            
            sub_p8 = valid_ly[(valid_ly['Frame'] >= p5_idx) & (valid_ly['Frame'] <= p5_idx + 60)]
            p8_idx = int(sub_p8.loc[sub_p8['LeftHandY_Smooth'].idxmax()]['Frame']) if not sub_p8.empty else p5_idx + 30
            
            valid_ry = df_db.dropna(subset=['RightHandY_Smooth'])
            sub_p12 = valid_ry[(valid_ry['Frame'] >= p8_idx) & (valid_ry['Frame'] <= total_frames - 10)]
            p12_idx = int(sub_p12.loc[sub_p12['RightHandY_Smooth'].idxmin()]['Frame']) if not sub_p12.empty else total_frames - 20
            p13_idx = total_frames - 1

            auto_f = {}
            auto_f["P1"] = p1_idx
            auto_f["P2"] = find_best_frame_from_db(df_db, 'ShaftAngle', 45.0, p1_idx, p5_idx)
            auto_f["P3"] = find_best_frame_from_db(df_db, 'ShaftAngle', 0.0, auto_f["P2"], p5_idx)
            auto_f["P4"] = find_best_frame_from_db(df_db, 'LtArmAngle', 0.0, auto_f["P3"], p5_idx)
            auto_f["P5"] = p5_idx
            
            auto_f["P6"] = find_best_frame_from_db(df_db, 'ShaftAngle', 315.0, p5_idx, p8_idx)
            auto_f["P7"] = find_best_frame_from_db(df_db, 'ShaftAngle', 0.0, auto_f["P6"], p8_idx)
            auto_f["P8"] = p8_idx
            
            auto_f["P9"] = find_best_frame_from_db(df_db, 'ShaftAngle', 135.0, p8_idx, p12_idx)
            auto_f["P10"] = find_best_frame_from_db(df_db, 'ShaftAngle', 180.0, auto_f["P9"], p12_idx)
            auto_f["P11"] = find_best_frame_from_db(df_db, 'RtArmAngle', 180.0, auto_f["P10"], p12_idx)
            auto_f["P12"] = p12_idx
            auto_f["P13"] = p13_idx

            st.session_state.auto_frames = auto_f
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps
            st.session_state.scan_done = True

    # 💡 다이나믹 DB 로드 렌더링 블록 (재탐색 없이 DB 좌표 즉시 호출 -> 0.1초 반응)
    if 'scan_done' in st.session_state:
        st.subheader("📸 DB 좌표 기반 실시간 다이나믹 미세조정 뷰")
        cols = st.columns(4)
        analysis_data = []
        fixed_ground = st.session_state.fixed_ground
        df_db = st.session_state.df_db

        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                phase_id = p['phase']
                auto_fn = st.session_state.auto_frames.get(phase_id, 0)
                
                # 슬라이더 조정
                fn = st.slider(f"[{phase_id}] 조정", 0, st.session_state.total_frames-1, auto_fn, key=f"slider_{phase_id}")
                
                img_path = os.path.join(st.session_state.frame_dir, f"frame_{fn:04d}.jpg")
                img = cv2.imread(img_path)
                
                # 💡 DB에서 해당 프레임의 데이터 추출
                frame_data = df_db[df_db['Frame'] == fn].iloc[0]
                
                measured_val = 0.0
                verification_status = "Pass"
                
                if img is not None:
                    cv2.line(img, fixed_ground[0], fixed_ground[1], (0, 0, 255), 4)
                    cv2.putText(img, "Fixed Ground", (fixed_ground[0][0], fixed_ground[0][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if p['type'] == 'shaft':
                        if not pd.isna(frame_data['WristX']) and not pd.isna(frame_data['TargetX']):
                            wx, wy = int(frame_data['WristX']), int(frame_data['WristY'])
                            tx, ty = int(frame_data['TargetX']), int(frame_data['TargetY'])
                            
                            cv2.circle(img, (wx, wy), 8, (0, 255, 255), -1)
                            cv2.circle(img, (tx, ty), 8, (0, 0, 255), -1)
                            cv2.line(img, (wx, wy), (tx, ty), (0, 255, 0), 4)
                            
                            measured_val = frame_data['ShaftAngle']
                            cv2.putText(img, f"Shaft: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    elif p['type'] == 'arm_left':
                        if not pd.isna(frame_data['LShoulderX']) and not pd.isna(frame_data['WristX']): # 손목 기준
                            lsx, lsy = int(frame_data['LShoulderX']), int(frame_data['LShoulderY'])
                            lwx, lwy = int(frame_data['WristX']), int(frame_data['WristY']) # 왼손목 대용
                            cv2.line(img, (lsx, lsy), (lwx, lwy), (0, 255, 0), 4)
                            measured_val = frame_data['LtArmAngle']
                            cv2.putText(img, f"Lt Arm: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    elif p['type'] == 'arm_right':
                        if not pd.isna(frame_data['RShoulderX']) and not pd.isna(frame_data['WristX']):
                            rsx, rsy = int(frame_data['RShoulderX']), int(frame_data['RShoulderY'])
                            rwx, rwy = int(frame_data['WristX']), int(frame_data['WristY'])
                            cv2.line(img, (rsx, rsy), (rwx, rwy), (0, 255, 0), 4)
                            measured_val = frame_data['RtArmAngle']
                            cv2.putText(img, f"Rt Arm: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if p['target_angle'] is not None and not pd.isna(measured_val):
                        error = angle_diff(measured_val, p['target_angle'])
                        if error > 20:
                            verification_status = "Check (Review)"

                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{phase_id}] {p['name']} ({verification_status})", use_column_width=True)
                
                analysis_data.append({
                    "Phase": phase_id,
                    "Name": p['name'],
                    "정의 기준 (Target)": p['desc'],
                    "목표 값": str(p['target_angle']) if p['target_angle'] is not None else "변곡점",
                    "AI 측정 값": measured_val,
                    "검증 상태": verification_status,
                    "Frame #": fn,
                    "Time Stamp(s)": round(fn / st.session_state.fps, 2)
                })

        st.divider()
        st.subheader("📊 좌표 DB 검증 결과 표")
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True, hide_index=True)
