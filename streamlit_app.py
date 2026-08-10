"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인 (변경 불가)]
1. 정적 배경 블랙리스트 (Static Background Rejection):
   - 고정된 카메라 환경에서, 여러 프레임에 걸쳐 좌표가 변하지 않는 객체는 배경 구조물(티 마커 등)로 
     간주하여 철저히 폐기함. 움직이는 동적 객체만 클럽 헤드로 추적. (대표님 인사이트)
2. P1 클럽 길이 동적 캘리브레이션 (P1 Club Length Boundary):
   - P1의 클럽 길이를 기준으로 반경을 설정하여 가짜 마커를 차단함.
3. 360도 스윙 벡터 타임라인 록인 (360° Timeline Lock-in):
   - Left=180°, Down=90°, Right=0°, Up=270° 체계를 수학적으로 완벽히 유지.
4. 다이나믹 렌더링 (Dynamic UI):
   - 프레임 미세조정 시 오버레이가 실시간 즉각 반영됨.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import os
import tempfile
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="P1-P13 Dynamic Master Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 정적 배경 차단 & 정밀 오버레이 시스템")
st.markdown("정지된 배경 마커를 원천 차단하고, 움직이는 진짜 클럽만을 추적하여 각도를 산출합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

phases_info = [
    {"phase": "P1", "name": "Address", "desc": "샤프트 지면 수직", "target_angle": 90.0, "type": "shaft"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트 45°", "target_angle": 45.0, "type": "shaft"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트 우측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔 우측 수평", "target_angle": 0.0, "type": "arm_left"},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (탑)", "target_angle": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "desc": "샤프트 다운스윙 315°", "target_angle": 315.0, "type": "shaft"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트 우측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P8", "name": "Impact", "desc": "볼 타격 (최저점)", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트 릴리스 135°", "target_angle": 135.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트 좌측 수평", "target_angle": 180.0, "type": "shaft"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔 좌측 수평", "target_angle": 180.0, "type": "arm_right"},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (피니시 진입)", "target_angle": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "desc": "스윙 종료 정지", "target_angle": None, "type": "finish"},
]

def calculate_peak_duration(y_coords, fps=30, threshold=10.0):
    valid_y = [y for y in y_coords if not np.isnan(y)]
    if not valid_y: return 0.0
    peak_y = min(valid_y) 
    return round(len([y for y in valid_y if abs(y - peak_y) <= threshold]) / fps, 3)

def compute_relative_angle(p1, p2, ground_p1, ground_p2):
    if ground_p1[0] > ground_p2[0]:
        ground_p1, ground_p2 = ground_p2, ground_p1
        
    g_dx = ground_p2[0] - ground_p1[0]
    g_dy = ground_p2[1] - ground_p1[1]
    ground_tilt = math.atan2(g_dy, g_dx)
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    cos_t = math.cos(-ground_tilt)
    sin_t = math.sin(-ground_tilt)
    rx = dx * cos_t - dy * sin_t
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
        with st.spinner("1단계: 정적 배경 학습 및 P1 클럽 길이 측정 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            
            p1_ground = None 
            ref_club_length = None
            
            # 💡 [핵심 방어 1] 정적 객체 식별을 위한 맵핑
            background_candidates = []
            
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
                
                if wrist_pt_first:
                    for box in c_res_first.boxes:
                        name = c_res_first.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if name in ['head', 'shaft']:
                            ref_club_length = math.hypot(cent[0] - wrist_pt_first[0], cent[1] - wrist_pt_first[1])
                            break
                if not ref_club_length:
                    ref_club_length = w_img * 0.3

            # 💡 [핵심 방어 2] 프레임을 순회하며 움직이지 않는 가짜 객체 탐지 (블랙리스트 생성)
            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if not ret or total_frames > 600: break
                
                c_res = custom_model(frame, verbose=False)[0]
                for box in c_res.boxes:
                    c = float(box.conf[0])
                    if c > 0.5: # 뚜렷하게 잡히는 객체 수집
                        name = c_res.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        background_candidates.append(cent)
                total_frames += 1
            temp_cap.release()

            # 💡 같은 자리에 20번 이상 찍힌 점은 배경 마커(티 마커 등)로 블랙리스트 등록
            static_blacklist = []
            for pt in background_candidates:
                count = sum(1 for p in background_candidates if math.hypot(p[0]-pt[0], p[1]-pt[1]) < 10)
                if count > 20: # 지속적으로 고정된 물체
                    # 이미 등록된 블랙리스트와 겹치지 않게 추가
                    if not any(math.hypot(p[0]-pt[0], p[1]-pt[1]) < 10 for p in static_blacklist):
                        static_blacklist.append(pt)
            
            st.session_state.fixed_ground = p1_ground
            st.session_state.ref_club_length = ref_club_length
            st.session_state.static_blacklist = static_blacklist
            st.session_state.max_allowed_dist = ref_club_length * 1.3 # 원근법 고려 1.3배 허용

        with st.spinner("2단계: 240장 전수 DB 구축 및 정적 배경 제거 중..."):
            db_records = []
            cap = cv2.VideoCapture(tfile.name)
            t_frames = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or t_frames > 600: break
                
                img_path = os.path.join(frame_dir, f"frame_{t_frames:04d}.jpg")
                cv2.imwrite(img_path, frame)
                
                analyzed_frame = cv2.imread(img_path)
                p_res = pose_model(analyzed_frame, verbose=False)[0]
                c_res = custom_model(analyzed_frame, verbose=False)[0]
                
                ly, ry, la, ra, sa = np.nan, np.nan, np.nan, np.nan, np.nan
                
                if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                    kpts = p_res.keypoints.xy[0].cpu().numpy()
                    conf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints.conf is not None else np.ones(len(kpts))
                    
                    if len(kpts) > 10:
                        if kpts[9][0] > 0 and conf[9] > 0.4: ly = kpts[9][1]
                        if kpts[10][0] > 0 and conf[10] > 0.4: ry = kpts[10][1]
                        
                        if kpts[5][0] > 0 and kpts[9][0] > 0:
                            la = compute_relative_angle((kpts[5][0], kpts[5][1]), (kpts[9][0], kpts[9][1]), p1_ground[0], p1_ground[1])
                        if kpts[6][0] > 0 and kpts[10][0] > 0:
                            ra = compute_relative_angle((kpts[6][0], kpts[6][1]), (kpts[10][0], kpts[10][1]), p1_ground[0], p1_ground[1])
                        
                        pts = []
                        if kpts[9][0] > 0 and conf[9] > 0.4: pts.append(kpts[9])
                        if kpts[10][0] > 0 and conf[10] > 0.4: pts.append(kpts[10])
                        
                        if pts:
                            wrist_pt = (int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts])))
                            valid_targets = []
                            
                            for box in c_res.boxes:
                                c = float(box.conf[0])
                                if c < 0.1: continue # 모션블러 감지
                                
                                name = c_res.names[int(box.cls[0])]
                                cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                                dist = math.hypot(cent[0] - wrist_pt[0], cent[1] - wrist_pt[1])
                                
                                # 💡 [핵심 방어 3] 블랙리스트(고정된 마커)에 속하는지 검사
                                is_background = any(math.hypot(cent[0]-bp[0], cent[1]-bp[1]) < 20 for bp in static_blacklist)
                                
                                # 고정된 가짜 객체가 아니고, 클럽 물리 반경 이내인 경우만 채택
                                if not is_background and dist < st.session_state.max_allowed_dist:
                                    valid_targets.append((cent, dist, c))
                            
                            if valid_targets:
                                target_pt = max(valid_targets, key=lambda x: x[1])[0] # 손목에서 가장 먼 진짜 클럽 끝
                                sa = compute_relative_angle(wrist_pt, target_pt, p1_ground[0], p1_ground[1])
                
                db_records.append({
                    "Frame": t_frames,
                    "LeftHandY": ly,
                    "RightHandY": ry,
                    "ShaftAngle": sa,
                    "LtArmAngle": la,
                    "RtArmAngle": ra
                })
                t_frames += 1
            cap.release()
            
            df_db = pd.DataFrame(db_records)
            df_db['LeftHandY_Smooth'] = df_db['LeftHandY'].rolling(window=5, min_periods=1, center=True).median()
            df_db['RightHandY_Smooth'] = df_db['RightHandY'].rolling(window=5, min_periods=1, center=True).median()
            st.session_state.df_db = df_db
            
            valid_ly = df_db.dropna(subset=['LeftHandY_Smooth'])
            p1_idx = int(valid_ly.iloc[0]['Frame']) if not valid_ly.empty else 0
            
            sub_p5 = valid_ly[valid_ly['Frame'] < t_frames * 0.5]
            p5_idx = int(sub_p5.loc[sub_p5['LeftHandY_Smooth'].idxmin()]['Frame']) if not sub_p5.empty else t_frames // 4
            
            sub_after_p5 = valid_ly[valid_ly['Frame'] > p5_idx].head(60)
            p8_idx = int(sub_after_p5.loc[sub_after_p5['LeftHandY_Smooth'].idxmax()]['Frame']) if not sub_after_p5.empty else p5_idx + 30
            
            valid_ry = df_db.dropna(subset=['RightHandY_Smooth'])
            sub_after_p8 = valid_ry[valid_ry['Frame'] > p8_idx]
            p12_idx = int(sub_after_p8.loc[sub_after_p8['RightHandY_Smooth'].idxmin()]['Frame']) if not sub_after_p8.empty else t_frames - 20
            
            sub_p13 = valid_ly[valid_ly['Frame'] > p12_idx]
            p13_idx = t_frames - 1
            for i in range(len(sub_p13) - 5):
                window = sub_p13.iloc[i:i+5]
                if window['LeftHandY_Smooth'].var() < 2.0:
                    p13_idx = int(window.iloc[0]['Frame'])
                    break

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

            st.session_state.p5_time = calculate_peak_duration(df_db['LeftHandY'][:p8_idx], fps)
            st.session_state.p12_time = calculate_peak_duration(df_db['RightHandY'][p8_idx:], fps)
            st.session_state.auto_frames = auto_f
            st.session_state.total_frames = t_frames
            st.session_state.fps = fps
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 정적 마커 필터링 기반 즉각 다이나믹 검증 뷰")
        cols = st.columns(4)
        analysis_data = []
        fixed_ground = st.session_state.fixed_ground
        max_allowed_dist = st.session_state.max_allowed_dist
        static_blacklist = st.session_state.static_blacklist

        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                phase_id = p['phase']
                auto_fn = st.session_state.auto_frames.get(phase_id, 0)
                
                # 💡 슬라이더 조작 시 아래 로직 실시간 실행 (미세조정 즉각 반영)
                fn = st.slider(f"[{phase_id}] 조정", 0, st.session_state.total_frames-1, auto_fn, key=f"slider_{phase_id}")
                
                img_path = os.path.join(st.session_state.frame_dir, f"frame_{fn:04d}.jpg")
                img = cv2.imread(img_path)
                
                measured_val = 0.0
                verification_status = "Pass"
                
                if img is not None:
                    p_res = pose_model(img, verbose=False)[0]
                    c_res = custom_model(img, verbose=False)[0]
                    kpts = p_res.keypoints.xy[0].cpu().numpy() if (p_res.keypoints is not None and len(p_res.keypoints.xy) > 0) else None
                    conf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints is not None and p_res.keypoints.conf is not None else np.ones(17)
                    
                    cv2.line(img, fixed_ground[0], fixed_ground[1], (0, 0, 255), 4)
                    cv2.putText(img, "Fixed Ground", (fixed_ground[0][0], fixed_ground[0][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    wrist_pt, target_pt = None, None
                    if kpts is not None and len(kpts) > 10:
                        pts = []
                        if kpts[9][0] > 0 and conf[9] > 0.4: pts.append(kpts[9])
                        if kpts[10][0] > 0 and conf[10] > 0.4: pts.append(kpts[10])
                        if pts: wrist_pt = (int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts])))
                    
                    if wrist_pt:
                        valid_targets = []
                        for box in c_res.boxes:
                            c = float(box.conf[0])
                            if c < 0.1: continue 
                            cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                            dist = math.hypot(cent[0] - wrist_pt[0], cent[1] - wrist_pt[1])
                            
                            # 블랙리스트에 걸리면 그리지 않음
                            is_background = any(math.hypot(cent[0]-bp[0], cent[1]-bp[1]) < 20 for bp in static_blacklist)
                            
                            if not is_background and dist < max_allowed_dist:
                                valid_targets.append((cent, dist, c))
                        
                        if valid_targets:
                            target_pt = max(valid_targets, key=lambda x: x[1])[0]

                    if p['type'] == 'shaft' and wrist_pt and target_pt:
                        cv2.circle(img, wrist_pt, 8, (0, 255, 255), -1)
                        cv2.circle(img, target_pt, 8, (0, 0, 255), -1)
                        cv2.line(img, wrist_pt, target_pt, (0, 255, 0), 4)
                        measured_val = compute_relative_angle(wrist_pt, target_pt, fixed_ground[0], fixed_ground[1])
                        cv2.putText(img, f"Shaft: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    elif 'arm' in p['type'] and kpts is not None and len(kpts) > 10:
                        s_idx = 5 if 'left' in p['type'] else 6
                        w_idx = 9 if 'left' in p['type'] else 10
                        if kpts[s_idx][0] > 0 and kpts[w_idx][0] > 0:
                            s_pt = (int(kpts[s_idx][0]), int(kpts[s_idx][1]))
                            w_pt = (int(kpts[w_idx][0]), int(kpts[w_idx][1]))
                            cv2.line(img, s_pt, w_pt, (0, 255, 0), 4)
                            measured_val = compute_relative_angle(s_pt, w_pt, fixed_ground[0], fixed_ground[1])
                            arm_label = "Rt Arm" if phase_id == "P11" else "Lt Arm"
                            cv2.putText(img, f"{arm_label}: {measured_val}deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if p['target_angle'] is not None:
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
        st.subheader("📊 정적 마커 블랙리스트 적용 검증 결과 표")
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드", data=csv_data,
            file_name='calibrated_swing_P1_P13.csv', mime='text/csv',
        )
