"""
================================================================================
[상용화 레벨: P1-P13 무결점 통합 마스터 엔진 (Sequence-Locked Anchor)]
1. 절대 무너지지 않는 뼈대: 어드레스 왜글(Waggle) 오인 방지를 위한 순차적 앵커링
2. 360도 수학적 완벽 동기화: Left=0°, Down=90°, Right=180°, Up=270°
3. 시각적 오버레이: UI 각도와 100% 일치하는 기하학적 Arc 및 0도 기준선 렌더링
4. 하이브리드 엔진: 100% 좌표 보간(Interpolation) + 라이브 추적(Live Tracking)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import os
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="P1-P13 Sequence-Locked Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 무결점 통합 정밀 분석 시스템")
st.markdown("어드레스 시의 흔들림을 임팩트로 오인하는 결함을 완벽히 해결한 최종 상용화 버전입니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

phases_info = [
    {"phase": "P1", "name": "Address", "desc": "샤프트 지면 수직", "target_angle": 90.0, "type": "shaft"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트 45°", "target_angle": 45.0, "type": "shaft"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트 좌측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔 좌측 수평", "target_angle": 0.0, "type": "arm_left"},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점", "target_angle": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "desc": "샤프트 315°", "target_angle": 315.0, "type": "shaft"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트 좌측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P8", "name": "Impact", "desc": "임팩트 (손목 최저점)", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트 135°", "target_angle": 135.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트 우측 수평", "target_angle": 180.0, "type": "shaft"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔 우측 수평", "target_angle": 180.0, "type": "arm_right"},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점", "target_angle": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "desc": "스윙 종료 정지", "target_angle": None, "type": "finish"},
]

def compute_relative_angle(p1, p2, ground_p1, ground_p2):
    if ground_p1[0] > ground_p2[0]:
        ground_p1, ground_p2 = ground_p2, ground_p1
    g_dx, g_dy = ground_p2[0] - ground_p1[0], ground_p2[1] - ground_p1[1]
    ground_tilt = math.atan2(g_dy, g_dx)
    
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    cos_t, sin_t = math.cos(-ground_tilt), math.sin(-ground_tilt)
    # Left=0, Down=90, Right=180, Up=270 체계 완벽 동기화 (X축 음수화)
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
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_angle_visual(img, vertex, target_pt, measured_val, ground_p1, ground_p2, color, label):
    if pd.isna(measured_val): return
    if ground_p1[0] > ground_p2[0]:
        ground_p1, ground_p2 = ground_p2, ground_p1
    
    g_dx, g_dy = ground_p2[0] - ground_p1[0], ground_p2[1] - ground_p1[1]
    ground_tilt_deg = math.degrees(math.atan2(g_dy, g_dx))
    
    # 0도 기준선 (좌측 방향)
    ref_rad = math.radians(180 + ground_tilt_deg)
    ref_x = int(vertex[0] + 80 * math.cos(ref_rad))
    ref_y = int(vertex[1] + 80 * math.sin(ref_rad))
    cv2.line(img, vertex, (ref_x, ref_y), (255, 255, 255), 2, cv2.LINE_AA)
    draw_text_with_outline(img, "0", (ref_x - 15, ref_y - 5), 0.5, (255, 255, 255), (0, 0, 0), 1)
    
    cv2.circle(img, vertex, 6, (0, 255, 255), -1)
    cv2.circle(img, target_pt, 6, (0, 0, 255), -1)
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    
    # 각도 아크 렌더링 (논리각도 -> 이미지 렌더링 각도 변환)
    pts = []
    num_steps = max(5, int(measured_val / 4))
    for i in range(num_steps + 1):
        a = i * (measured_val / num_steps)
        a_vis = 180 - a + ground_tilt_deg
        a_img_rad = math.radians(a_vis)
        px = vertex[0] + 45 * math.cos(a_img_rad)
        py = vertex[1] + 45 * math.sin(a_img_rad)
        pts.append([int(px), int(py)])
    
    if pts: cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
        
    mid_a = measured_val / 2.0 if measured_val > 0 else 0
    mid_a_img_rad = math.radians(180 - mid_a + ground_tilt_deg)
    txt_x = int(vertex[0] + 65 * math.cos(mid_a_img_rad))
    txt_y = int(vertex[1] + 65 * math.sin(mid_a_img_rad))
    draw_text_with_outline(img, f"{label}: {measured_val}deg", (txt_x - 30, txt_y + 10), 0.7, (0, 255, 255), (0, 0, 0), 2)

uploaded_file = st.file_uploader("스윙 영상 업로드", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    if 'scan_done' not in st.session_state:
        with st.spinner("1단계: 정적 배경 학습 및 P1 물리 캘리브레이션 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            p1_ground, p1_head, ref_club_length = None, None, None
            background_candidates = []
            
            ret, first_frame = cap.read()
            if ret:
                h_img, w_img, _ = first_frame.shape
                p_res_first = pose_model(first_frame, verbose=False)[0]
                c_res_first = custom_model(first_frame, verbose=False)[0]
                
                wrist_pt_first = None
                if p_res_first.keypoints is not None and len(p_res_first.keypoints.xy) > 0:
                    kpts_f = p_res_first.keypoints.xy[0].cpu().numpy()
                    if len(kpts_f) >= 5:
                        head_pts = [kpts_f[i] for i in range(5) if kpts_f[i][0] > 0]
                        if head_pts: p1_head = (int(np.mean([p[0] for p in head_pts])), int(np.mean([p[1] for p in head_pts])))
                    if len(kpts_f) > 16 and kpts_f[15][0] > 0 and kpts_f[16][0] > 0:
                        p1_ground = ((int(kpts_f[15][0]), int(kpts_f[15][1])), (int(kpts_f[16][0]), int(kpts_f[16][1])))
                    if len(kpts_f) > 10 and kpts_f[9][0] > 0 and kpts_f[10][0] > 0:
                        wrist_pt_first = (int((kpts_f[9][0]+kpts_f[10][0])/2), int((kpts_f[9][1]+kpts_f[10][1])/2))
                
                if not p1_ground: p1_ground = ((int(w_img * 0.35), int(h_img * 0.85)), (int(w_img * 0.65), int(h_img * 0.85)))
                
                if wrist_pt_first:
                    for box in c_res_first.boxes:
                        if c_res_first.names[int(box.cls[0])] in ['head', 'shaft']:
                            cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                            ref_club_length = math.hypot(cent[0] - wrist_pt_first[0], cent[1] - wrist_pt_first[1])
                            break
                if not ref_club_length: ref_club_length = w_img * 0.3

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                c_res = custom_model(frame, verbose=False)[0]
                for box in c_res.boxes:
                    if float(box.conf[0]) > 0.4:
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if wrist_pt_first and math.hypot(cent[0]-wrist_pt_first[0], cent[1]-wrist_pt_first[1]) > (ref_club_length * 0.5):
                            background_candidates.append(cent)
            
            static_blacklist = []
            for pt in background_candidates:
                if sum(1 for p in background_candidates if math.hypot(p[0]-pt[0], p[1]-pt[1]) < 15) > (total_frames * 0.1): 
                    if not any(math.hypot(p[0]-pt[0], p[1]-pt[1]) < 15 for p in static_blacklist):
                        static_blacklist.append(pt)
            
            st.session_state.fixed_ground = p1_ground
            st.session_state.max_allowed_dist = ref_club_length * 1.3
            st.session_state.static_blacklist = static_blacklist
            st.session_state.p1_head = p1_head

        with st.spinner("2단계: 전수 좌표 분석 및 100% 보간(Interpolation) 중..."):
            db_records = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            t_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                cv2.imwrite(os.path.join(frame_dir, f"frame_{t_frames:04d}.jpg"), frame)
                
                p_res = pose_model(frame, verbose=False)[0]
                c_res = custom_model(frame, verbose=False)[0]
                
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
                            wx, wy = int(kpts[9][0]), int(kpts[9][1])
                            la = compute_relative_angle((lsx, lsy), (wx, wy), p1_ground[0], p1_ground[1])
                        if kpts[6][0] > 0 and kpts[10][0] > 0:
                            rsx, rsy = int(kpts[6][0]), int(kpts[6][1])
                            rwx, rwy = int(kpts[10][0]), int(kpts[10][1])
                            ra = compute_relative_angle((rsx, rsy), (rwx, rwy), p1_ground[0], p1_ground[1])
                        
                        if not pd.isna(wx):
                            valid_targets = []
                            for box in c_res.boxes:
                                if float(box.conf[0]) < 0.1: continue 
                                cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                                dist = math.hypot(cent[0] - wx, cent[1] - wy)
                                is_bg = any(math.hypot(cent[0]-bp[0], cent[1]-bp[1]) < 20 for bp in static_blacklist) if t_frames >= 5 else False
                                if not is_bg and dist < st.session_state.max_allowed_dist:
                                    valid_targets.append((cent, dist))
                            
                            if valid_targets:
                                target_pt = max(valid_targets, key=lambda p: p[1])[0] 
                                tx, ty = target_pt[0], target_pt[1]
                                sa = compute_relative_angle((wx, wy), target_pt, p1_ground[0], p1_ground[1])
                
                db_records.append({
                    "Frame": t_frames, "LeftHandY": ly, "RightHandY": ry,
                    "ShaftAngle": sa, "LtArmAngle": la, "RtArmAngle": ra,
                    "WristX": wx, "WristY": wy, "TargetX": tx, "TargetY": ty,
                    "LShoulderX": lsx, "LShoulderY": lsy, "RShoulderX": rsx, "RShoulderY": rsy
                })
                t_frames += 1
            cap.release()
            
            # 결측치 100% 선형 보간
            df_db = pd.DataFrame(db_records).interpolate(method='linear', limit_direction='both')
            df_db['WristY_Smooth'] = df_db['WristY'].rolling(window=9, min_periods=1, center=True).mean()
            st.session_state.df_db = df_db

        with st.spinner("3단계: 시퀀스-락(Sequence-Lock) 타임라인 구축 중..."):
            # 💡 [핵심] 앵커 붕괴 방지를 위한 시퀀스 락(Sequence-Lock) 알고리즘
            # 1. P5(Top): 영상의 첫 65% 이내에서 손목이 가장 높은(Y 최소) 지점
            search_end_p5 = int(t_frames * 0.65)
            p5_idx = int(df_db['WristY_Smooth'].iloc[:search_end_p5].idxmin())
            
            # 2. P12(Finish): P5 이후 20프레임 시점부터 가장 높은(Y 최소) 지점
            p12_idx = int(df_db['WristY_Smooth'].iloc[p5_idx + 20:].idxmin()) if len(df_db.iloc[p5_idx + 20:]) > 0 else t_frames - 1
            
            # 3. P8(Impact): 무조건 P5와 P12 사이에서 가장 낮은(Y 최대) 지점 (어드레스 왜글 완벽 배제)
            sub_imp = df_db['WristY_Smooth'].iloc[p5_idx + 5 : p12_idx - 5]
            p8_idx = int(sub_imp.idxmax()) if not sub_imp.empty else p5_idx + (p12_idx - p5_idx) // 2
            
            # 4. P1(Address): P5 이전에서 손이 가장 낮았던(Y 최대) 지점
            sub_add = df_db['WristY_Smooth'].iloc[:max(1, p5_idx - 5)]
            p1_idx = int(sub_add.idxmax()) if not sub_add.empty else 0
            
            p13_idx = t_frames - 1

            # 타임라인 절대 록인
            auto_f = {"P1": p1_idx, "P5": p5_idx, "P8": p8_idx, "P12": p12_idx, "P13": p13_idx}
            auto_f["P2"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 45.0, p1_idx+1, p5_idx-3)
            auto_f["P3"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 0.0, auto_f["P2"]+1, p5_idx-2)
            auto_f["P4"] = find_strictly_bounded_frame(df_db, 'LtArmAngle', 0.0, auto_f["P3"]+1, p5_idx-1)
            auto_f["P6"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 315.0, p5_idx+1, p8_idx-2)
            auto_f["P7"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 0.0, auto_f["P6"]+1, p8_idx-1)
            auto_f["P9"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 135.0, p8_idx+1, p12_idx-3)
            auto_f["P10"] = find_strictly_bounded_frame(df_db, 'ShaftAngle', 180.0, auto_f["P9"]+1, p12_idx-2)
            auto_f["P11"] = find_strictly_bounded_frame(df_db, 'RtArmAngle', 180.0, auto_f["P10"]+1, p12_idx-1)

            st.session_state.auto_frames = auto_f
            st.session_state.total_frames = t_frames
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 무결점 타임라인 & 하이브리드 오버레이 뷰")
        cols = st.columns(4)
        analysis_data = []
        fixed_ground = st.session_state.fixed_ground
        max_dist = st.session_state.max_allowed_dist
        static_blacklist = st.session_state.static_blacklist
        p1_head = st.session_state.p1_head 
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
                    used_live = False
                    wrist_pt, target_pt, ls_pt, rs_pt = None, None, None, None
                    
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
                            if float(box.conf[0]) < 0.1: continue 
                            cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                            dist = math.hypot(cent[0] - wrist_pt[0], cent[1] - wrist_pt[1])
                            is_bg = any(math.hypot(cent[0]-bp[0], cent[1]-bp[1]) < 20 for bp in static_blacklist) if fn >= 5 else False
                            if not is_bg and dist < max_dist: valid_targets.append((cent, dist))
                        if valid_targets:
                            target_pt = max(valid_targets, key=lambda x: x[1])[0] 
                            used_live = True

                    elif 'arm' in p['type'] and wrist_pt and ((p['type'] == 'arm_left' and ls_pt) or (p['type'] == 'arm_right' and rs_pt)):
                        used_live = True

                    # 💡 라이브 실패 시 완벽히 보간된 DB 좌표로 즉각 폴백(Fallback)
                    if not used_live:
                        wrist_pt = (int(frame_data['WristX']), int(frame_data['WristY'])) if not pd.isna(frame_data['WristX']) else None
                        target_pt = (int(frame_data['TargetX']), int(frame_data['TargetY'])) if not pd.isna(frame_data['TargetX']) else None
                        ls_pt = (int(frame_data['LShoulderX']), int(frame_data['LShoulderY'])) if not pd.isna(frame_data['LShoulderX']) else None
                        rs_pt = (int(frame_data['RShoulderX']), int(frame_data['RShoulderY'])) if not pd.isna(frame_data['RShoulderX']) else None

                    if p['type'] == 'shaft' and wrist_pt and target_pt:
                        measured_val = compute_relative_angle(wrist_pt, target_pt, fixed_ground[0], fixed_ground[1])
                        draw_angle_visual(img, wrist_pt, target_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Shaft")
                    elif p['type'] == 'arm_left' and ls_pt and wrist_pt:
                        measured_val = compute_relative_angle(ls_pt, wrist_pt, fixed_ground[0], fixed_ground[1])
                        draw_angle_visual(img, ls_pt, wrist_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Lt Arm")
                    elif p['type'] == 'arm_right' and rs_pt and wrist_pt:
                        measured_val = compute_relative_angle(rs_pt, wrist_pt, fixed_ground[0], fixed_ground[1])
                        draw_angle_visual(img, rs_pt, wrist_pt, measured_val, fixed_ground[0], fixed_ground[1], (0, 255, 0), "Rt Arm")

                    if p['target_angle'] is not None and not pd.isna(measured_val):
                        if angle_diff(measured_val, p['target_angle']) > 7.0: verification_status = "Check (Review)"
                        
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{phase_id}] {p['name']} ({verification_status})", use_column_width=True)
                
                analysis_data.append({
                    "Phase": phase_id, "Name": p['name'], "Target": p['target_angle'] if p['target_angle'] is not None else "변곡점",
                    "Measured": measured_val, "Status": verification_status, "Frame": fn
                })

        st.divider()
        st.subheader("📊 절대 순서 보장 하이브리드 분석 결과 (Tolerance: 7°)")
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True, hide_index=True)
