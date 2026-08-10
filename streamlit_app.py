"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인 (변경 불가)]
1. 머리 축 고정 시각화 (Head Sway Tracking):
   - P1 시점의 골퍼 머리(안면 랜드마크 평균점) 좌표를 저장하고, 노란색 기준원을 렌더링.
2. 임팩트 정밀도 극대화 (Impact Precision):
   - P8 탐지 시, 왼손 좌표가 아닌 '클럽 헤드(TargetY)'가 가장 지면에 가까워진 
     최저점(Max Y) 프레임을 추출하여 1프레임의 오차도 없는 임팩트를 잡아냄.
3. 라이브 다이나믹 오버레이 (Live Dynamic Overlay):
   - 미세조정 시 AI가 실시간으로 샤프트를 추적하며, 직관적인 회전 호(Arc)를 그려줌.
4. 가짜 객체 철통 방어 (Static & Boundary Filter):
   - 정지된 배경 마커와 물리적 한계를 벗어난 가짜 클럽을 원천 차단함.
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

st.set_page_config(page_title="P1-P13 Ultimate Master Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 스웨이 감지 & 정밀 임팩트 분석 시스템")
st.markdown("머리 축(Head Sway) 고정 여부 확인과 클럽 최저점 기반의 초정밀 임팩트 탐지를 제공합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

phases_info = [
    {"phase": "P1", "name": "Address", "desc": "샤프트 지면 수직", "target_angle": 90.0, "type": "shaft"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트 45°", "target_angle": 45.0, "type": "shaft"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트 좌측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔 좌측 수평", "target_angle": 0.0, "type": "arm_left"},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (탑)", "target_angle": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "desc": "샤프트 다운스윙 315°", "target_angle": 315.0, "type": "shaft"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트 좌측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P8", "name": "Impact", "desc": "볼 타격 (클럽 최저점)", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트 릴리스 135°", "target_angle": 135.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트 우측 수평", "target_angle": 180.0, "type": "shaft"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔 우측 수평", "target_angle": 180.0, "type": "arm_right"},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (피니시 진입)", "target_angle": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "desc": "스윙 종료 정지", "target_angle": None, "type": "finish"},
]

def calculate_peak_duration(y_coords, fps=30, threshold=10.0):
    valid_y = [y for y in y_coords if not pd.isna(y)]
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
    rx = -(dx * cos_t - dy * sin_t)
    ry = dx * sin_t + dy * cos_t
    
    angle = math.degrees(math.atan2(ry, rx))
    if angle < 0: angle += 360
    return round(angle, 1)

def angle_diff(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def find_strictly_bounded_frame(db_df, col_name, target_val, start_f, end_f):
    if start_f >= end_f: return start_f
    sub = db_df[(db_df['Frame'] >= start_f) & (db_df['Frame'] <= end_f)]
    valid = sub.dropna(subset=[col_name])
    if valid.empty: return start_f
    diffs = valid[col_name].apply(lambda x: angle_diff(x, target_val))
    return int(valid.loc[diffs.idxmin()]['Frame'])

def draw_text_with_outline(img, text, pos, font_scale, text_color, outline_color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_angle_visual(img, vertex, target_pt, measured_val, ground_p1, ground_p2, color, label):
    if pd.isna(measured_val): return
    
    if ground_p1[0] > ground_p2[0]:
        ground_p1, ground_p2 = ground_p2, ground_p1
        
    g_dx = ground_p2[0] - ground_p1[0]
    g_dy = ground_p2[1] - ground_p1[1]
    ground_tilt_deg = math.degrees(math.atan2(g_dy, g_dx))
    
    ref_rad = math.radians(180 + ground_tilt_deg)
    ref_x = int(vertex[0] + 80 * math.cos(ref_rad))
    ref_y = int(vertex[1] + 80 * math.sin(ref_rad))
    cv2.line(img, vertex, (ref_x, ref_y), (255, 255, 255), 2, cv2.LINE_AA)
    draw_text_with_outline(img, "0", (ref_x - 15, ref_y - 5), 0.5, (255, 255, 255), (0, 0, 0), 1)
    
    cv2.circle(img, vertex, 6, (0, 255, 255), -1)
    cv2.circle(img, target_pt, 6, (0, 0, 255), -1)
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    
    pts = []
    num_steps = max(5, int(measured_val / 4))
    for i in range(num_steps + 1):
        a = i * (measured_val / num_steps)
        a_img_rad = math.radians(180 - a + ground_tilt_deg)
        px = vertex[0] + 45 * math.cos(a_img_rad)
        py = vertex[1] + 45 * math.sin(a_img_rad)
        pts.append([int(px), int(py)])
    
    if pts:
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
        
    mid_a = measured_val / 2.0 if measured_val > 0 else 0
    mid_a_img_rad = math.radians(180 - mid_a + ground_tilt_deg)
    txt_x = int(vertex[0] + 65 * math.cos(mid_a_img_rad))
    txt_y = int(vertex[1] + 65 * math.sin(mid_a_img_rad))
    
    text_str = f"{label}: {measured_val}deg"
    draw_text_with_outline(img, text_str, (txt_x - 40, txt_y + 5), 0.7, (0, 255, 255), (0, 0, 0), 2)

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    if 'auto_frames' not in st.session_state:
        with st.spinner("1단계: 정적 배경 학습 및 P1 머리/클럽 캘리브레이션 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            
            p1_ground = None 
            ref_club_length = None
            p1_golfer_head = None # 💡 P1 골퍼 머리 좌표 저장용
            
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
                    
                    # 💡 [1번 기능] P1 골퍼의 머리 중심 좌표 추출 (안면 랜드마크 0~4번 평균)
                    if len(kpts_f) >= 5:
                        head_pts = [kpts_f[i] for i in range(5) if kpts_f[i][0] > 0]
                        if head_pts:
                            hx = int(np.mean([p[0] for p in head_pts]))
                            hy = int(np.mean([p[1] for p in head_pts]))
                            p1_golfer_head = (hx, hy)

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

            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if not ret or total_frames > 600: break
                c_res = custom_model(frame, verbose=False)[0]
                for box in c_res.boxes:
                    c = float(box.conf[0])
                    if c > 0.4:
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if wrist_pt_first and math.hypot(cent[0]-wrist_pt_first[0], cent[1]-wrist_pt_first[1]) > (ref_club_length * 0.5):
                            background_candidates.append(cent)
                total_frames += 1
            temp_cap.release()

            static_blacklist = []
            for pt in background_candidates:
                count = sum(1 for p in background_candidates if math.hypot(p[0]-pt[0], p[1]-pt[1]) < 15)
                if count > 30: 
                    if not any(math.hypot(p[0]-pt[0], p[1]-pt[1]) < 15 for p in static_blacklist):
                        static_blacklist.append(pt)
            
            st.session_state.fixed_ground = p1_ground
            st.session_state.max_allowed_dist = ref_club_length * 1.3
            st.session_state.static_blacklist = static_blacklist
            st.session_state.p1_golfer_head = p1_golfer_head # 머리 좌표 세션 저장

        with st.spinner("2단계: 240장 전수 DB 추출 및 클럽 최저점(임팩트) 판별 중..."):
            db_records = []
            cap = cv2.VideoCapture(tfile.name)
            t_frames = 0
            prev_tx, prev_ty = None, None 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or t_frames > 600: break
                
                img_path = os.path.join(frame_dir, f"frame_{t_frames:04d}.jpg")
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
                        if kpts[9][0] > 0 and conf[9] > 0.1: ly = kpts[9][1]
                        if kpts[10][0] > 0 and conf[10] > 0.1: ry = kpts[10][1]
                        
                        if kpts[5][0] > 0 and kpts[9][0] > 0:
                            lsx, lsy = int(kpts[5][0]), int(kpts[5][1])
                            lwx, lwy = int(kpts[9][0]), int(kpts[9][1])
                            la = compute_relative_angle((lsx, lsy), (lwx, lwy), p1_ground[0], p1_ground[1])
                        if kpts[6][0] > 0 and kpts[10][0] > 0:
                            rsx, rsy = int(kpts[6][0]), int(kpts[6][1])
                            rwx, rwy = int(kpts[10][0]), int(kpts[10][1])
                            ra = compute_relative_angle((rsx, rsy), (rwx, rwy), p1_ground[0], p1_ground[1])
                        
                        pts = []
                        if kpts[9][0] > 0 and conf[9] > 0.1: pts.append(kpts[9])
                        if kpts[10][0] > 0 and conf[10] > 0.1: pts.append(kpts[10])
                        
                        if pts:
                            wrist_pt = (int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts])))
                            wx, wy = wrist_pt[0], wrist_pt[1]
                            valid_targets = []
                            
                            for box in c_res.boxes:
                                c = float(box.conf[0])
                                if c < 0.1: continue 
                                
                                cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                                dist = math.hypot(cent[0] - wrist_pt[0], cent[1] - wrist_pt[1])
                                
                                is_bg = any(math.hypot(cent[0]-bp[0], cent[1]-bp[1]) < 20 for bp in static_blacklist)
                                if t_frames < 5: is_bg = False 
                                
                                if not is_bg and dist < st.session_state.max_allowed_dist:
                                    valid_targets.append((cent, dist))
                            
                            if valid_targets:
                                if prev_tx is not None and prev_ty is not None:
                                    target_pt = min(valid_targets, key=lambda p: math.hypot(p[0][0]-prev_tx, p[0][1]-prev_ty))[0]
                                else:
                                    target_pt = max(valid_targets, key=lambda p: p[1])[0] 
                                
                                prev_tx, prev_ty = target_pt[0], target_pt[1]
                                tx, ty = target_pt[0], target_pt[1]
                                sa = compute_relative_angle(wrist_pt, target_pt, p1_ground[0], p1_ground[1])
                            else:
                                prev_tx, prev_ty = None, None
                
                db_records.append({
                    "Frame": t_frames,
                    "LeftHandY": ly, "RightHandY": ry,
                    "ShaftAngle": sa, "LtArmAngle": la, "RtArmAngle": ra,
                    "WristX": wx, "WristY": wy, "TargetX": tx, "TargetY": ty,
                    "LShoulderX": lsx, "LShoulderY": lsy, "RShoulderX": rsx, "RShoulderY": rsy
                })
                t_frames += 1
            cap.release()
            
            df_db = pd.DataFrame(db_records)
            
            # 모든 좌표 선형 보간 (결측치 메우기)
            cols_to_interpolate = ['LeftHandY', 'RightHandY', 'ShaftAngle', 'LtArmAngle', 'RtArmAngle',
                                   'WristX', 'WristY', 'TargetX', 'TargetY', 
                                   'LShoulderX', 'LShoulderY', 'RShoulderX', 'RShoulderY']
            for col in cols_to_interpolate:
                df_db[col] = df_db[col].interpolate(method='linear', limit_direction='both')
                
            df_db['LeftHandY_Smooth'] = df_db['LeftHandY'].rolling(window=5, min_periods=1, center=True).median()
            df_db['RightHandY_Smooth'] = df_db['RightHandY'].rolling(window=5, min_periods=1, center=True).median()
            
            # 💡 [3번 기능] 임팩트 타격점 탐지를 위해 클럽 헤드(TargetY) 스무딩 추가
            df_db['TargetY_Smooth'] = df_db['TargetY'].rolling(window=3, min_periods=1, center=True).mean()
            
            st.session_state.df_db = df_db
            
            p1_idx = 0
            
            sub_p5 = df_db[(df_db['Frame'] >= p1_idx) & (df_db['Frame'] <= t_frames * 0.6)]
            p5_idx = int(sub_p5.loc[sub_p5['LeftHandY_Smooth'].idxmin()]['Frame']) if not sub_p5.empty else t_frames // 4
            
            sub_p12 = df_db.dropna(subset=['RightHandY_Smooth'])
            sub_p12 = sub_p12[(sub_p12['Frame'] >= p5_idx + 20) & (sub_p12['Frame'] <= t_frames - 10)]
            p12_idx = int(sub_p12.loc[sub_p12['RightHandY_Smooth'].idxmin()]['Frame']) if not sub_p12.empty else t_frames - 20
            
            # 💡 [3번 기능] P8(임팩트) 정밀 탐지: 손목이 아닌 '클럽 헤드(TargetY)'가 가장 아래로 내려온 최저점 프레임!
            sub_p8 = df_db[(df_db['Frame'] > p5_idx) & (df_db['Frame'] < p12_idx)]
            if not sub_p8.empty:
                p8_idx = int(sub_p8.loc[sub_p8['TargetY_Smooth'].idxmax()]['Frame'])
            else:
                p8_idx = p5_idx + 30
            
            p13_idx = t_frames - 1

            # 타임라인 절대 순서 강제 매핑
            auto_f = {}
            auto_f["P1"] = p1_idx
            auto_f["P5"] = p5_idx
            auto_f["P8"] = p8_idx
            auto_f["P12"] = p12_idx
            auto_f["P13"] = p13_idx

            auto_f["P2"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 45.0, p1_idx+1, p5_idx-3)
            auto_f["P3"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 0.0, auto_f["P2"]+1, p5_idx-2)
            auto_f["P4"] = find_strictly_bounded_frame(df_db, 'LtArmAngle', 0.0, auto_f["P3"]+1, p5_idx-1)
            
            auto_f["P6"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 315.0, p5_idx+1, p8_idx-2)
            auto_f["P7"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 0.0, auto_f["P6"]+1, p8_idx-1)
            
            auto_f["P9"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 135.0, p8_idx+1, p12_idx-3)
            auto_f["P10"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 180.0, auto_f["P9"]+1, p12_idx-2)
            auto_f["P11"] = find_strictly_bounded_frame(df_db, 'RtArmAngle', 180.0, auto_f["P10"]+1, p12_idx-1)

            st.session_state.p5_time = calculate_peak_duration(df_db['LeftHandY'][:p8_idx], fps)
            st.session_state.p12_time = calculate_peak_duration(df_db['RightHandY'][p8_idx:], fps)
            st.session_state.auto_frames = auto_f
            st.session_state.total_frames = t_frames
            st.session_state.fps = fps
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 머리축 고정(Sway) 확인 & 라이브 다이나믹 오버레이 뷰")
        cols = st.columns(4)
        analysis_data = []
        fixed_ground = st.session_state.fixed_ground
        max_dist = st.session_state.max_allowed_dist
        static_blacklist = st.session_state.static_blacklist
        p1_head = st.session_state.p1_golfer_head # 💡 P1 머리 좌표

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
                    # 💡 라이브 AI 재구동 (미세조정 시 샤프트 찰떡같이 따라가기)
                    p_res = pose_model(img, verbose=False)[0]
                    c_res = custom_model(img, verbose=False)[0]
                    kpts = p_res.keypoints.xy[0].cpu().numpy() if (p_res.keypoints is not None and len(p_res.keypoints.xy) > 0) else None
                    conf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints is not None and p_res.keypoints.conf is not None else np.ones(17)
                    
                    cv2.line(img, fixed_ground[0], fixed_ground[1], (0, 0, 255), 4)
                    draw_text_with_outline(img, "Fixed Ground", (fixed_ground[0][0], fixed_ground[0][1]+30), 0.6, (0, 0, 255), (255, 255, 255), 2)
                    
                    # 💡 [1번 기능] 머리 축 고정 확인을 위한 P1 Reference Circle 렌더링
                    if p1_head:
                        hx, hy = p1_head
                        radius = int(max_dist * 0.12) # 적당한 머리 반경 크기
                        cv2.circle(img, (hx, hy), radius, (0, 255, 255), 2, cv2.LINE_AA) # 노란색 머리 축 가이드라인
                        draw_text_with_outline(img, "Head(P1)", (hx - 35, hy - radius - 10), 0.6, (0, 255, 255), (0, 0, 0), 1)
                    
                    wrist_pt, target_pt = None, None
                    if kpts is not None and len(kpts) > 10:
                        pts = []
                        if kpts[9][0] > 0 and conf[9] > 0.1: pts.append(kpts[9])
                        if kpts[10][0] > 0 and conf[10] > 0.1: pts.append(kpts[10])
                        if pts: wrist_pt = (int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts])))
                    
                    if p['type'] == 'shaft' and wrist_pt:
                        valid_targets = []
                        for box in c_res.boxes:
                            c = float(box.conf[0])
                            if c < 0.1: continue 
                            cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                            dist = math.hypot(cent[0] - wrist_pt[0], cent[1] - wrist_pt[1])
                            
                            is_bg = any(math.hypot(cent[0]-bp[0], cent[1]-bp[1]) < 20 for bp in static_blacklist)
                            if fn < 5: is_bg = False
                            
                            if not is_bg and dist < max_dist:
                                valid_targets.append((cent, dist))
                        
                        if valid_targets:
                            target_pt = max(valid_targets, key=lambda x: x[1])[0] 
                            measured_val = compute_relative_angle(wrist_pt, target_pt, fixed_ground[0], fixed_ground[1])
                            draw_angle_visual(img, wrist_pt, target_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Shaft")
                    
                    elif p['type'] == 'arm_left' and kpts is not None and len(kpts) > 10:
                        if kpts[5][0] > 0 and kpts[9][0] > 0:
                            s_pt = (int(kpts[5][0]), int(kpts[5][1]))
                            w_pt = (int(kpts[9][0]), int(kpts[9][1]))
                            measured_val = compute_relative_angle(s_pt, w_pt, fixed_ground[0], fixed_ground[1])
                            draw_angle_visual(img, s_pt, w_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Lt Arm")

                    elif p['type'] == 'arm_right' and kpts is not None and len(kpts) > 10:
                        if kpts[6][0] > 0 and kpts[10][0] > 0:
                            s_pt = (int(kpts[6][0]), int(kpts[6][1]))
                            w_pt = (int(kpts[10][0]), int(kpts[10][1]))
                            measured_val = compute_relative_angle(s_pt, w_pt, fixed_ground[0], fixed_ground[1])
                            draw_angle_visual(img, s_pt, w_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Rt Arm")

                    if p['target_angle'] is not None and not pd.isna(measured_val) and measured_val != 0.0:
                        error = angle_diff(measured_val, p['target_angle'])
                        if error > 7.0:
                            verification_status = "Check (Review)"
                        else:
                            verification_status = "Pass"

                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{phase_id}] {p['name']} ({verification_status})", use_column_width=True)
                
                analysis_data.append({
                    "Phase": phase_id,
                    "Name": p['name'],
                    "정의 기준 (Target)": p['desc'],
                    "목표 값": str(p['target_angle']) if p['target_angle'] is not None else "변곡점",
                    "AI 측정 값": measured_val if measured_val != 0.0 else "N/A",
                    "검증 상태": verification_status,
                    "Frame #": fn,
                    "Time Stamp(s)": round(fn / st.session_state.fps, 2)
                })

        st.divider()
        st.subheader("📊 헤드 축 고정 검증 및 타임라인 결과 표")
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True, hide_index=True)
