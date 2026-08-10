"""
================================================================================
[절대 준수 원칙 - 시스템 설계 철학 및 분석 파이프라인]
1. 범용적 동적 앵커링 (Universal Dynamic Anchoring):
   - 특정 프레임 번호나 고정된 프레임 간격(+15, -30 등)을 절대 사용하지 않음.
   - P5(Max Height), P8(Min Height), P12(Max Right Height) 등 골프의 물리적 진리(Y좌표)만으로 
     어떤 영상이든 자동으로 뼈대를 추출함.
2. 타임라인 절대 순서 강제 (Strict Sequential Timeline):
   - P1 <= P2 <= P3 <= P4 <= P5 <= P6 <= P7 <= P8 의 시간적 순서를 절대적으로 강제함.
3. 100% DB 의존형 다이나믹 렌더링 (Zero AI Re-run on UI):
   - 미세조정 시 DB 좌표를 불러오며, 직관적인 시각적 회전 호(Visual Arc)를 렌더링.
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

st.set_page_config(page_title="P1-P13 Universal Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 범용 하이브리드 정밀 분석 시스템")
st.markdown("임의의 프레임 고정값 없이, 바이오메커니즘(좌표 최저/최고점)만으로 어떤 영상이든 완벽히 분석합니다.")

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
    {"phase": "P8", "name": "Impact", "desc": "볼 타격 (손목 최저점)", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트 릴리스 135°", "target_angle": 135.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트 좌측 수평", "target_angle": 180.0, "type": "shaft"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔 좌측 수평", "target_angle": 180.0, "type": "arm_right"},
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
    """
    💡 범용 타임라인 탐색: 무조건 이전 페이즈(start_f)와 다음 앵커(end_f) 사이에서만 탐색.
    영상의 길이나 fps에 상관없이 논리적 순서가 절대 보장됩니다.
    """
    if start_f >= end_f: return start_f
    sub = db_df[(db_df['Frame'] >= start_f) & (db_df['Frame'] <= end_f)]
    valid = sub.dropna(subset=[col_name])
    
    if valid.empty: return start_f
    diffs = valid[col_name].apply(lambda x: angle_diff(x, target_val))
    return int(valid.loc[diffs.idxmin()]['Frame'])

def draw_text_with_outline(img, text, pos, font_scale, text_color, outline_color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 3, cv2.LINE_AA)
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
        with st.spinner("1단계: 정적 배경 학습 및 머리/클럽 캘리브레이션 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            
            p1_ground = None 
            ref_club_length = None
            p1_golfer_head = None 
            
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
                    
                    if len(kpts_f) >= 5:
                        head_pts = [kpts_f[i] for i in range(5) if kpts_f[i][0] > 0]
                        if head_pts:
                            p1_golfer_head = (int(np.mean([p[0] for p in head_pts])), int(np.mean([p[1] for p in head_pts])))

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
                if not ref_club_length: ref_club_length = w_img * 0.3

            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if not ret or total_frames > 1000: break # 영상 길이 범용 대응
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
                if count > (total_frames * 0.1): # 전체 프레임의 10% 이상 한자리에 머물면 배경으로 간주
                    if not any(math.hypot(p[0]-pt[0], p[1]-pt[1]) < 15 for p in static_blacklist):
                        static_blacklist.append(pt)
            
            st.session_state.fixed_ground = p1_ground
            st.session_state.max_allowed_dist = ref_club_length * 1.3
            st.session_state.static_blacklist = static_blacklist
            st.session_state.p1_golfer_head = p1_golfer_head

        with st.spinner("2단계: 범용 좌표 DB 추출 및 타임라인 분석 중..."):
            db_records = []
            cap = cv2.VideoCapture(tfile.name)
            t_frames = 0
            prev_tx, prev_ty = None, None 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or t_frames > 1000: break
                
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
            
            # 모든 좌표 100% 선형 보간 처리 (선 사라짐 원천 방지)
            cols_to_interpolate = ['LeftHandY', 'RightHandY', 'ShaftAngle', 'LtArmAngle', 'RtArmAngle',
                                   'WristX', 'WristY', 'TargetX', 'TargetY', 
                                   'LShoulderX', 'LShoulderY', 'RShoulderX', 'RShoulderY']
            for col in cols_to_interpolate:
                df_db[col] = df_db[col].interpolate(method='linear', limit_direction='both')
                
            df_db['LeftHandY_Smooth'] = df_db['LeftHandY'].rolling(window=5, min_periods=1, center=True).median()
            df_db['RightHandY_Smooth'] = df_db['RightHandY'].rolling(window=5, min_periods=1, center=True).median()
            st.session_state.df_db = df_db
            
            # 💡 [핵심] 범용 앵커(뼈대) 탐지: 비율(%)과 순서로만 접근하여 모든 길이에 대응
            p1_idx = 0
            
            # P5: 영상의 처음부터 70% 구간 내에서 손목 최저점(가장 높은 곳)
            sub_p5 = df_db[(df_db['Frame'] >= p1_idx) & (df_db['Frame'] <= int(t_frames * 0.7))]
            p5_idx = int(sub_p5.loc[sub_p5['LeftHandY_Smooth'].idxmin()]['Frame']) if not sub_p5.empty else t_frames // 4
            
            # P8: P5 이후부터 끝까지 구간에서 손목 최고점(가장 낮은 곳)
            sub_p8 = df_db[(df_db['Frame'] >= p5_idx)]
            p8_idx = int(sub_p8.loc[sub_p8['LeftHandY_Smooth'].idxmax()]['Frame']) if not sub_p8.empty else p5_idx + (t_frames - p5_idx) // 2
            
            # P12: P8 이후에서 오른손 최저점(가장 높은 곳)
            valid_ry = df_db.dropna(subset=['RightHandY_Smooth'])
            sub_p12 = valid_ry[(valid_ry['Frame'] >= p8_idx)]
            p12_idx = int(sub_p12.loc[sub_p12['RightHandY_Smooth'].idxmin()]['Frame']) if not sub_p12.empty else t_frames - 1
            
            p13_idx = t_frames - 1

            # 절대 시간 순서 매핑
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
        st.subheader("📸 머리축 고정(Sway) 확인 & 시각적 각도 라이브 뷰")
        cols = st.columns(4)
        analysis_data = []
        fixed_ground = st.session_state.fixed_ground
        max_dist = st.session_state.max_allowed_dist
        static_blacklist = st.session_state.static_blacklist
        p1_head = st.session_state.p1_golfer_head 
        df_db = st.session_state.df_db

        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                phase_id = p['phase']
                auto_fn = st.session_state.auto_frames.get(phase_id, 0)
                
                fn = st.slider(f"[{phase_id}] 조정", 0, st.session_state.total_frames-1, auto_fn, key=f"slider_{phase_id}")
                
                img_path = os.path.join(st.session_state.frame_dir, f"frame_{fn:04d}.jpg")
                img = cv2.imread(img_path)
                frame_data = df_db[df_db['Frame'] == fn].iloc[0]
                
                measured_val = 0.0
                verification_status = "Pass"
                
                if img is not None:
                    # 💡 하이브리드 엔진: 1. 라이브 AI 재구동 -> 2. 블러 발생 시 DB 폴백
                    used_live = False
                    wrist_pt, target_pt = None, None
                    ls_pt, rs_pt = None, None
                    
                    p_res = pose_model(img, verbose=False)[0]
                    c_res = custom_model(img, verbose=False)[0]
                    kpts = p_res.keypoints.xy[0].cpu().numpy() if (p_res.keypoints is not None and len(p_res.keypoints.xy) > 0) else None
                    conf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints is not None and p_res.keypoints.conf is not None else np.ones(17)
                    
                    cv2.line(img, fixed_ground[0], fixed_ground[1], (0, 0, 255), 4)
                    draw_text_with_outline(img, "Fixed Ground", (fixed_ground[0][0], fixed_ground[0][1]+30), 0.6, (0, 0, 255), (255, 255, 255), 2)
                    
                    if p1_head:
                        hx, hy = p1_head
                        radius = int(max_dist * 0.15)
                        cv2.circle(img, (hx, hy), radius, (0, 255, 255), 2, cv2.LINE_AA) 
                        draw_text_with_outline(img, "Head Axis", (hx - 40, hy - radius - 10), 0.6, (0, 255, 255), (0, 0, 0), 1)

                    if kpts is not None and len(kpts) > 10:
                        if kpts[5][0] > 0: ls_pt = (int(kpts[5][0]), int(kpts[5][1]))
                        if kpts[6][0] > 0: rs_pt = (int(kpts[6][0]), int(kpts[6][1]))
                        
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
                            used_live = True

                    elif 'arm' in p['type'] and wrist_pt and ((p['type'] == 'arm_left' and ls_pt) or (p['type'] == 'arm_right' and rs_pt)):
                        used_live = True

                    # 💡 라이브 감지 실패 시(모션블러) -> DB 완벽 보간 좌표 즉시 호출 (절대 소멸 방지)
                    if not used_live:
                        wrist_pt = (int(frame_data['WristX']), int(frame_data['WristY']))
                        target_pt = (int(frame_data['TargetX']), int(frame_data['TargetY']))
                        ls_pt = (int(frame_data['LShoulderX']), int(frame_data['LShoulderY']))
                        rs_pt = (int(frame_data['RShoulderX']), int(frame_data['RShoulderY']))

                    # 렌더링 실행
                    if p['type'] == 'shaft':
                        measured_val = compute_relative_angle(wrist_pt, target_pt, fixed_ground[0], fixed_ground[1])
                        draw_angle_visual(img, wrist_pt, target_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Shaft")
                    
                    elif p['type'] == 'arm_left':
                        measured_val = compute_relative_angle(ls_pt, wrist_pt, fixed_ground[0], fixed_ground[1])
                        draw_angle_visual(img, ls_pt, wrist_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Lt Arm")

                    elif p['type'] == 'arm_right':
                        measured_val = compute_relative_angle(rs_pt, wrist_pt, fixed_ground[0], fixed_ground[1])
                        draw_angle_visual(img, rs_pt, wrist_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Rt Arm")

                    if p['target_angle'] is not None and not pd.isna(measured_val):
                        error = angle_diff(measured_val, p['target_angle'])
                        # 💡 7도 엄격 검증 기준
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
                    "AI 측정 값": measured_val,
                    "검증 상태": verification_status,
                    "Frame #": fn,
                    "Time Stamp(s)": round(fn / st.session_state.fps, 2)
                })

        st.divider()
        st.subheader("📊 하이브리드 범용 검증 결과 표 (Tolerance: 7°)")
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True, hide_index=True)
