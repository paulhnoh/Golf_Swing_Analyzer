"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인 (변경 불가)]
1. 240장 전수 스캔 및 DB 구축 (Full Frame-by-Frame DB):
   - 모든 프레임을 개별 이미지로 저장하고, 가상 지면선을 기준으로 각도를 전수 계산함.
2. 360도 스윙 벡터 각도계 (360° Circular Angle System):
   - Left=0°, Down=90°, Right=180°, Up=270° 체계를 도입하여 각 페이즈의 방향성을 완벽히 분리.
   - P3/P7(0°), P6(315°), P9(135°), P10(180°) 등 스윙 흐름에 맞춘 기하학적 각도 할당.
3. 타임라인 기반 구간 한정 검색 (Timeline Bounded Search):
   - P1 ➔ P5(탑) ➔ P8(임팩트) ➔ P12(피니시 진입) 대구간을 선행 탐지하고, 
     그 범위 내에서만 원형 각도 오차(Circular Diff)가 가장 적은 프레임을 추출.
4. P1 고정 가상 지면선 (Fixed Virtual Ground):
   - P1 시점의 양발목 좌표를 전체 프레임의 영구적인 수평 기준으로 삼음.
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

st.set_page_config(page_title="P1-P13 Master Vector Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 마스터 벡터 정밀 분석 시스템")
st.markdown("대표님의 360° 스윙 벡터 모델을 도입하여 P1~P13 싯점을 완벽히 매핑합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

# 대표님의 다이어그램에 완벽하게 일치시킨 목표 각도 체계
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
    """
    대표님의 좌표계 구현 (Left=0, Down=90, Right=180, Up=270)
    지면선의 기울기를 보정한 후, 해당 체계로 각도를 산출합니다.
    """
    # X좌표 정렬 (항상 왼쪽에서 오른쪽으로 향하는 지면 벡터 생성)
    if ground_p1[0] > ground_p2[0]:
        ground_p1, ground_p2 = ground_p2, ground_p1
        
    g_dx = ground_p2[0] - ground_p1[0]
    g_dy = ground_p2[1] - ground_p1[1]
    ground_tilt = math.atan2(g_dy, g_dx)
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # 지면 기울기(Tilt)만큼 회전시켜 수평을 맞춤
    cos_t = math.cos(-ground_tilt)
    sin_t = math.sin(-ground_tilt)
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t
    
    # Left(0), Down(90), Right(180), Up(270) 각도 체계 적용
    angle = math.degrees(math.atan2(ry, -rx))
    if angle < 0: angle += 360
    return round(angle, 1)

def angle_diff(a, b):
    """원형 각도 체계의 최소 오차 산출 (예: 359도와 1도의 차이는 2도)"""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def find_best_frame_from_db(db_df, col_name, target_val, start_f, end_f):
    """지정된 타임라인 구간 내에서 목표 각도와 가장 오차가 적은 프레임을 추출"""
    sub = db_df[(db_df['Frame'] >= start_f) & (db_df['Frame'] <= end_f)]
    if sub.empty: return start_f
    valid = sub.dropna(subset=[col_name])
    if valid.empty: return start_f
    
    # 원형 오차(angle_diff)를 적용하여 최소 오차 행 탐색
    diffs = valid[col_name].apply(lambda x: angle_diff(x, target_val))
    best_row = valid.loc[diffs.idxmin()]
    return int(best_row['Frame'])

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    if 'auto_frames' not in st.session_state:
        with st.spinner("240장 전수 DB 구축 및 벡터 각도 탐색 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            
            db_records = []
            p1_ground = None 
            
            # 1. P1 고정 지면선 확보
            temp_cap = cv2.VideoCapture(tfile.name)
            ret, first_frame = temp_cap.read()
            if ret:
                h_img, w_img, _ = first_frame.shape
                p_res_first = pose_model(first_frame, verbose=False)[0]
                if p_res_first.keypoints is not None and len(p_res_first.keypoints.xy) > 0:
                    kpts_f = p_res_first.keypoints.xy[0].cpu().numpy()
                    if len(kpts_f) > 16 and kpts_f[15][0] > 0 and kpts_f[16][0] > 0:
                        p1_ground = ((int(kpts_f[15][0]), int(kpts_f[15][1])), (int(kpts_f[16][0]), int(kpts_f[16][1])))
                if not p1_ground:
                    p1_ground = ((int(w_img * 0.35), int(h_img * 0.85)), (int(w_img * 0.65), int(h_img * 0.85)))
            temp_cap.release()
            st.session_state.fixed_ground = p1_ground

            # 2. 240장 전수 스캔 및 DB(Dataframe) 저장
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or total_frames > 600: break
                
                img_path = os.path.join(frame_dir, f"frame_{total_frames:04d}.jpg")
                cv2.imwrite(img_path, frame)
                
                analyzed_frame = cv2.imread(img_path)
                p_res = pose_model(analyzed_frame, verbose=False)[0]
                c_res = custom_model(analyzed_frame, verbose=False)[0]
                
                ly, ry, la, ra, sa = np.nan, np.nan, np.nan, np.nan, np.nan
                
                if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                    kpts = p_res.keypoints.xy[0].cpu().numpy()
                    if len(kpts) > 10:
                        if kpts[9][0] > 0: ly = kpts[9][1]
                        if kpts[10][0] > 0: ry = kpts[10][1]
                        
                        # 팔 각도 (어깨 -> 손목)
                        if kpts[5][0] > 0 and kpts[9][0] > 0:
                            la = compute_relative_angle((kpts[5][0], kpts[5][1]), (kpts[9][0], kpts[9][1]), p1_ground[0], p1_ground[1])
                        if kpts[6][0] > 0 and kpts[10][0] > 0:
                            ra = compute_relative_angle((kpts[6][0], kpts[6][1]), (kpts[10][0], kpts[10][1]), p1_ground[0], p1_ground[1])
                        
                        # 샤프트 각도 (양손 중앙 -> 클럽 헤드/샤프트)
                        wrist_pt = (int((kpts[9][0]+kpts[10][0])/2), int((kpts[9][1]+kpts[10][1])/2)) if (kpts[9][0] > 0 and kpts[10][0] > 0) else None
                        if wrist_pt:
                            head, shaft = None, None
                            for box in c_res.boxes:
                                name = c_res.names[int(box.cls[0])]
                                cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                                if name == 'shaft': shaft = cent
                                elif name == 'head': head = cent
                            target_pt = shaft if shaft else head
                            if target_pt:
                                sa = compute_relative_angle(wrist_pt, target_pt, p1_ground[0], p1_ground[1])
                
                db_records.append({
                    "Frame": total_frames,
                    "LeftHandY": ly,
                    "RightHandY": ry,
                    "ShaftAngle": sa,
                    "LtArmAngle": la,
                    "RtArmAngle": ra
                })
                total_frames += 1
            cap.release()
            
            df_db = pd.DataFrame(db_records)
            st.session_state.df_db = df_db
            
            # 3. 뼈대(Anchor) 구간 설정
            valid_ly = df_db.dropna(subset=['LeftHandY'])
            p1_idx = int(valid_ly.iloc[0]['Frame']) if not valid_ly.empty else 0
            p5_idx = int(valid_ly.loc[valid_ly['LeftHandY'].idxmin()]['Frame']) if not valid_ly.empty else total_frames // 4
            
            sub_ry = df_db.iloc[p5_idx:].dropna(subset=['RightHandY'])
            p12_idx = int(sub_ry.loc[sub_ry['RightHandY'].idxmin()]['Frame']) if not sub_ry.empty else total_frames - 30
            p13_idx = total_frames - 1
            
            sub_impact = df_db.loc[p5_idx:p12_idx].dropna(subset=['LeftHandY'])
            p8_idx = int(sub_impact.loc[sub_impact['LeftHandY'].idxmax()]['Frame']) if not sub_impact.empty else (p5_idx + p12_idx) // 2

            # 4. 각 페이즈별 정밀 검색 (순서 역전 방지)
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
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 벡터 각도 기반 정밀 오버레이 검증 뷰")
        cols = st.columns(4)
        analysis_data = []
        fixed_ground = st.session_state.fixed_ground

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
                    kpts = p_res.keypoints.xy[0].cpu().numpy() if (p_res.keypoints is not None and len(p_res.keypoints.xy) > 0) else None
                    
                    cv2.line(img, fixed_ground[0], fixed_ground[1], (0, 0, 255), 4)
                    cv2.putText(img, "Fixed Ground (P1)", (fixed_ground[0][0], fixed_ground[0][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    wrist_pt, target_pt = None, None
                    if kpts is not None and len(kpts) > 10 and kpts[9][0] > 0 and kpts[10][0] > 0:
                        wrist_pt = (int((kpts[9][0]+kpts[10][0])/2), int((kpts[9][1]+kpts[10][1])/2))
                    
                    head, shaft = None, None
                    for box in c_res.boxes:
                        name = c_res.names[int(box.cls[0])]
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if name == 'shaft': shaft = cent
                        elif name == 'head': head = cent
                    target_pt = shaft if shaft else head

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
                        # 원형 오차(angle_diff)로 검증 수행
                        error = angle_diff(measured_val, p['target_angle'])
                        if error > 20: # 허용 오차 20도
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
        st.subheader("📊 벡터 궤적 분석 및 정밀 검증 결과 표")
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드", data=csv_data,
            file_name='calibrated_swing_P1_P13.csv', mime='text/csv',
        )
