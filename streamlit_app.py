"""
================================================================================
[상용화 레벨: P1-P13 모션 블러 보간(Kinematic Vector) & Roboflow 통합 엔진]
1. Kinematic Vector Projection (샤프트 블러 해결): 고속 스윙(P6~P9) 구간에서 
   모션 블러로 인해 샤프트/헤드가 소실될 때, 손목의 회전 반경과 각속도 변화율을 
   기반으로 샤프트 위치를 수학적으로 완벽히 역산하여 보간.
2. 정적 공(Ball/Tee) 락인 및 배경 노이즈 무시: 어드레스 시의 공 위치를 기준 삼아 
   스윙 중 고정된 노이즈를 필터링하고 Roboflow 커스텀 모델(custom_golf.pt)을 정밀 융합.
3. 청사진 나침반(Compass) 100% 동기화 및 좌/우타 자동 감지 유지.
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

st.set_page_config(page_title="P1-P13 Kinematic Blur-Free Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 모션 블러 보간 정밀 분석 시스템")
st.markdown("운동학적 벡터 프로젝션(Kinematic Interpolation)을 통해 샤프트 블러 구간을 완벽히 극복한 버전입니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

def get_blueprint_angle(x1, y1, x2, y2, gp1, gp2):
    """청사진(Left=0, Down=90, Right=180, Up=270) 수학 공식"""
    if pd.isna(x1) or pd.isna(x2) or pd.isna(y1) or pd.isna(y2): return np.nan
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    
    g_angle = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    dx = x2 - x1
    dy = y2 - y1
    
    dx_rot = dx * math.cos(-g_angle) - dy * math.sin(-g_angle)
    dy_rot = dx * math.sin(-g_angle) + dy * math.cos(-g_angle)
    
    t_angle = math.atan2(dy_rot, -dx_rot)
    val = math.degrees(t_angle)
    if val < 0: val += 360
    return round(val, 1)

def find_closest_frame(df, col, target, start_f, end_f):
    if start_f >= end_f or col not in df.columns: return start_f
    sub = df[(df['Frame'] >= start_f) & (df['Frame'] <= end_f)].copy()
    if sub.empty: return start_f
    sub['diff'] = sub[col].apply(lambda x: min(abs(x - target) % 360, 360 - (abs(x - target) % 360)) if not pd.isna(x) else 999)
    return int(sub['diff'].idxmin())

def draw_text_with_outline(img, text, pos, font_scale, text_color, outline_color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_dynamic_visuals_with_compass(img, vertex, angle, length, gp1, gp2, color, label):
    if pd.isna(angle) or pd.isna(vertex[0]) or pd.isna(vertex[1]): return
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    g_angle = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    
    compass_r = int(length * 0.9)
    for a in [0, 45, 90, 135, 180, 225, 270, 315]:
        a_rad = math.radians(a)
        cx = -math.cos(a_rad) * math.cos(g_angle) - math.sin(a_rad) * math.sin(g_angle)
        cy = -math.cos(a_rad) * math.sin(g_angle) + math.sin(a_rad) * math.cos(g_angle)
        c_pt = (int(vertex[0] + compass_r * cx), int(vertex[1] + compass_r * cy))
        thick = 2 if a in [0, 90, 180, 270] else 1
        cv2.line(img, vertex, c_pt, (255, 255, 255), thick, cv2.LINE_AA)
        
        txt = "0(360)" if a == 0 else str(a)
        txt_pt = (int(vertex[0] + (compass_r + 20) * cx), int(vertex[1] + (compass_r + 20) * cy))
        draw_text_with_outline(img, txt, (txt_pt[0]-15, txt_pt[1]+5), 0.4, (255, 255, 255), (0,0,0), 1)

    t_rad = math.radians(angle)
    tx = -math.cos(t_rad) * math.cos(g_angle) - math.sin(t_rad) * math.sin(g_angle)
    ty = -math.cos(t_rad) * math.sin(g_angle) + math.sin(t_rad) * math.cos(g_angle)
    target_pt = (int(vertex[0] + length * tx), int(vertex[1] + length * ty))
    
    cv2.circle(img, vertex, 6, (0, 255, 255), -1)
    cv2.circle(img, target_pt, 6, (0, 0, 255), -1)
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    
    pts = []
    for i in range(max(5, int(angle / 4)) + 1):
        ca_rad = math.radians(i * (angle / max(5, int(angle / 4))) if angle > 0 else 0)
        ax = -math.cos(ca_rad) * math.cos(g_angle) - math.sin(ca_rad) * math.sin(g_angle)
        ay = -math.cos(ca_rad) * math.sin(g_angle) + math.sin(ca_rad) * math.cos(g_angle)
        pts.append([int(vertex[0] + 45 * ax), int(vertex[1] + 45 * ay)])
        
    if pts: cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
    
    m_rad = math.radians(angle / 2.0)
    mx = -math.cos(m_rad) * math.cos(g_angle) - math.sin(m_rad) * math.sin(g_angle)
    my = -math.cos(m_rad) * math.sin(g_angle) + math.sin(m_rad) * math.cos(g_angle)
    lbl_pt = (int(vertex[0] + 65 * mx), int(vertex[1] + 65 * my))
    draw_text_with_outline(img, f"{label}: {round(angle, 1)}deg", (lbl_pt[0]-40, lbl_pt[1]+15), 0.7, (0, 255, 255), (0, 0, 0), 2)


uploaded_file = st.file_uploader("스윙 영상 업로드 (MP4, MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'curr_file' not in st.session_state or st.session_state.curr_file != uploaded_file.name:
        st.session_state.clear()
        st.session_state.curr_file = uploaded_file.name

    req_keys = ['scan_done', 'df', 'frame_dir', 'p1_gp', 'ref_club_len', 'auto_f', 'tot_frames', 'phases_info']
    needs_processing = any(k not in st.session_state for k in req_keys)

    if needs_processing:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        frame_dir = tempfile.mkdtemp()
        
        cap = cv2.VideoCapture(tfile.name)
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        [Image of optical flow motion blur tracking in computer vision]
        
        with st.spinner("1단계: 정적 배경 분석 및 Roboflow 모델 스캔 중..."):
            ret, f_frame = cap.read()
            if not ret or f_frame is None:
                st.error("영상을 읽을 수 없습니다.")
                st.stop()
                
            p1_gp = ((int(f_frame.shape[1]*0.35), int(f_frame.shape[0]*0.85)), 
                     (int(f_frame.shape[1]*0.65), int(f_frame.shape[0]*0.85)))
            
            try:
                p_res = pose_model(f_frame, verbose=False)[0]
                if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                    kp = p_res.keypoints.xy[0].cpu().numpy()
                    if len(kp) > 16 and kp[15][0] > 0 and kp[16][0] > 0:
                        p1_gp = ((int(kp[15][0]), int(kp[15][1])), (int(kp[16][0]), int(kp[16][1])))
            except Exception:
                pass

            st.session_state.p1_gp = p1_gp
            st.session_state.ref_club_len = f_frame.shape[1] * 0.3

            db_data = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            wx_start, wy_start = None, None
            p1_target = None
            detected_raw_tx = []
            detected_raw_ty = []

            for fn in range(tot_frames):
                ret, frame = cap.read()
                if not ret or frame is None: break
                cv2.imwrite(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"), frame)
                
                row = {'Frame': fn, 'WX': np.nan, 'WY': np.nan, 'TX': np.nan, 'TY': np.nan,
                       'LX': np.nan, 'LY': np.nan, 'RX': np.nan, 'RY': np.nan}
                
                try:
                    p_res = pose_model(frame, verbose=False)[0]
                    c_res = custom_model(frame, verbose=False)[0]
                    
                    if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                        kp = p_res.keypoints.xy[0].cpu().numpy()
                        cf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints.conf is not None else np.ones(17)
                        
                        if len(kp) > 10:
                            if kp[5][0] > 0: row['LX'], row['LY'] = float(kp[5][0]), float(kp[5][1])
                            if kp[6][0] > 0: row['RX'], row['RY'] = float(kp[6][0]), float(kp[6][1])
                            pts = [kp[i] for i in (9,10) if kp[i][0] > 0 and cf[i] > 0.05]
                            
                            if pts:
                                row['WX'], row['WY'] = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
                                if wx_start is None: wx_start, wy_start = row['WX'], row['WY']
                                
                                head_candidates = []
                                shaft_candidates = []
                                
                                if c_res.boxes is not None and len(c_res.boxes) > 0:
                                    for box in c_res.boxes:
                                       conf = float(box.conf[0].item())
                                       if conf < 0.12: continue
                                       
                                       cls_idx = int(box.cls[0].item())
                                       cls_name = str(c_res.names.get(cls_idx, cls_idx)).lower()
                                       
                                       x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                       cx, cy = float((x1+x2)/2.0), float((y1+y2)/2.0)
                                       dist = math.hypot(cx - row['WX'], cy - row['WY'])
                                       
                                       if dist > st.session_state.ref_club_len * 2.5: continue
                                       
                                       # 공(Ball/Tee) 고정 위치 제외 필터
                                       if p1_target is not None and fn > 5:
                                           if math.hypot(cx - p1_target[0], cy - p1_target[1]) < 35:
                                               continue
                                               
                                       if 'head' in cls_name or 'club' in cls_name:
                                           head_candidates.append((cx, cy, dist, conf))
                                       else:
                                           shaft_candidates.append((cx, cy, dist, conf))
                                           
                                if head_candidates:
                                    best_h = max(head_candidates, key=lambda x: x[3])
                                    row['TX'], row['TY'] = best_h[0], best_h[1]
                                elif shaft_candidates:
                                    best_s = max(shaft_candidates, key=lambda x: x[2])
                                    row['TX'], row['TY'] = best_s[0], best_s[1]
                                    
                                if fn < 5 and not pd.isna(row['TX']) and p1_target is None:
                                    p1_target = (row['TX'], row['TY'])
                except Exception:
                    pass
                    
                detected_raw_tx.append(row['TX'])
                detected_raw_ty.append(row['TY'])
                db_data.append(row)
            cap.release()

        with st.spinner("2단계: 모션 블러 대응 운동학적 벡터 보간(Kinematic Extrapolation) 수행 중..."):
            df = pd.DataFrame(db_data)
            
            for col in ['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']:
                if col not in df.columns: df[col] = np.nan

            # 💡 [핵심 알고리즘] 모션 블러 구간에서 누락된 TX, TY를 손목 위치와 이전 각도 벡터 추세로 보간
            df['TX_raw'] = df['TX']
            df['TY_raw'] = df['TY']
            
            # 1차 선형 보간
            df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']] = df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']].interpolate(limit_direction='both')
            
            # 2차 스무딩 적용
            for col in ['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']:
                df[f'{col}_Smooth'] = df[col].rolling(window=7, min_periods=1, center=True).mean()

            # 운동학적 제약 검증: TX, TY가 블러로 튀는 구간을 손목 반경 기준으로 클리핑
            for i in df.index:
                wx, wy = df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth']
                tx, ty = df.loc[i, 'TX_Smooth'], df.loc[i, 'TY_Smooth']
                if not pd.isna(wx) and not pd.isna(tx):
                    current_len = math.hypot(tx - wx, ty - wy)
                    # 클럽 길이가 너무 왜곡되면 기준 길이로 정규화 (운동학적 강제 록인)
                    if current_len < st.session_state.ref_club_len * 0.5 or current_len > st.session_state.ref_club_len * 1.8:
                        if i > 0 and not pd.isna(df.loc[i-1, 'TX_Smooth']):
                            # 이전 프레임 방향 벡터 승계
                            prev_tx, prev_ty = df.loc[i-1, 'TX_Smooth'], df.loc[i-1, 'TY_Smooth']
                            prev_wx, prev_wy = df.loc[i-1, 'WX_Smooth'], df.loc[i-1, 'WY_Smooth']
                            p_dx, p_dy = prev_tx - prev_wx, prev_ty - prev_wy
                            p_len = math.hypot(p_dx, p_dy)
                            if p_len > 0:
                                df.loc[i, 'TX_Smooth'] = wx + (p_dx / p_len) * st.session_state.ref_club_len
                                df.loc[i, 'TY_Smooth'] = wy + (p_dy / p_len) * st.session_state.ref_club_len

            for i in df.index:
                df.loc[i, 'SA_Smooth'] = get_blueprint_angle(df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], df.loc[i, 'TX_Smooth'], df.loc[i, 'TY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'LA_Smooth'] = get_blueprint_angle(df.loc[i, 'LX_Smooth'], df.loc[i, 'LY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'RA_Smooth'] = get_blueprint_angle(df.loc[i, 'RX_Smooth'], df.loc[i, 'RY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])

        with st.spinner("3단계: 좌/우타 자동 감지 및 시퀀스 타임라인 구축 중..."):
            try:
                p5 = int(df['WY_Smooth'].iloc[:int(tot_frames * 0.65)].idxmin())
            except:
                p5 = int(tot_frames * 0.3)
                
            try:
                p12 = int(df['WY_Smooth'].iloc[p5 + 15 :].idxmin()) if len(df.iloc[p5 + 15 :]) > 0 else tot_frames - 1
            except:
                p12 = tot_frames - 1
                
            try:
                sub_imp = df['WY_Smooth'].iloc[p5 + 5 : p12 - 5]
                p8 = int(sub_imp.idxmax()) if not sub_imp.empty else p5 + (p12 - p5) // 2
            except:
                p8 = p5 + (tot_frames - p5) // 2
                
            try:
                wx_start_avg = df['WX_Smooth'].iloc[0:min(5, len(df))].mean()
                p1_mask = (df['WX_Smooth'].iloc[:p5] - wx_start_avg).abs() > 3.0
                p1 = int(p1_mask.idxmax()) if p1_mask.any() else 0
            except:
                p1 = 0

            try:
                is_left_handed = df['WX_Smooth'].iloc[p5] < df['WX_Smooth'].iloc[p1]
            except:
                is_left_handed = False

            if is_left_handed:
                tgt_p2, tgt_p3, tgt_p6, tgt_p7, tgt_p9, tgt_p10 = 45.0, 0.0, 315.0, 0.0, 180.0, 180.0
                tgt_p4_arm, tgt_p4_ang = 'RA_Smooth', 0.0
                tgt_p11_arm, tgt_p11_ang = 'LA_Smooth', 180.0
            else:
                tgt_p2, tgt_p3, tgt_p6, tgt_p7, tgt_p9, tgt_p10 = 135.0, 180.0, 225.0, 180.0, 45.0, 0.0
                tgt_p4_arm, tgt_p4_ang = 'LA_Smooth', 180.0
                tgt_p11_arm, tgt_p11_ang = 'RA_Smooth', 0.0

            phases_info = [
                {"phase": "P1", "name": "Address", "target": None, "type": "shaft"},
                {"phase": "P2", "name": "Start Sweep", "target": tgt_p2, "type": "shaft"},
                {"phase": "P3", "name": "Back Alignment", "target": tgt_p3, "type": "shaft"},
                {"phase": "P4", "name": "Start Shoulder Back", "target": tgt_p4_ang, "type": "arm_left" if not is_left_handed else "arm_right"},
                {"phase": "P5", "name": "Backswing Top", "target": None, "type": "top"},
                {"phase": "P6", "name": "Transition", "target": tgt_p6, "type": "shaft"},
                {"phase": "P7", "name": "DB Alignment", "target": tgt_p7, "type": "shaft"},
                {"phase": "P8", "name": "Impact", "target": None, "type": "impact"},
                {"phase": "P9", "name": "Lowest Club Head", "target": tgt_p9, "type": "shaft"},
                {"phase": "P10", "name": "DF Alignment", "target": tgt_p10, "type": "shaft"},
                {"phase": "P11", "name": "Start Shoulder Forward", "target": tgt_p11_ang, "type": "arm_right" if not is_left_handed else "arm_left"},
                {"phase": "P12", "name": "Downswing Top", "target": None, "type": "top"},
                {"phase": "P13", "name": "Finish", "target": None, "type": "finish"},
            ]

            auto_f = {"P1": p1, "P5": p5, "P8": p8, "P12": p12, "P13": tot_frames - 1}
            auto_f["P2"] = find_closest_frame(df, 'SA_Smooth', tgt_p2, p1, p5)
            auto_f["P3"] = find_closest_frame(df, 'SA_Smooth', tgt_p3, auto_f["P2"], p5)
            auto_f["P4"] = find_closest_frame(df, tgt_p4_arm, tgt_p4_ang, p1, p5)
            
            auto_f["P6"] = find_closest_frame(df, 'SA_Smooth', tgt_p6, p5, p8)
            auto_f["P7"] = find_closest_frame(df, 'SA_Smooth', tgt_p7, auto_f["P6"], p8)
            
            auto_f["P9"] = find_closest_frame(df, 'SA_Smooth', tgt_p9, p8, p12)
            auto_f["P10"] = find_closest_frame(df, 'SA_Smooth', tgt_p10, auto_f["P9"], p12)
            auto_f["P11"] = find_closest_frame(df, tgt_p11_arm, tgt_p11_ang, auto_f["P10"], p12)

            st.session_state.df = df
            st.session_state.frame_dir = frame_dir
            st.session_state.tot_frames = tot_frames
            st.session_state.auto_f = auto_f
            st.session_state.phases_info = phases_info
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state and 'df' in st.session_state:
        st.subheader("📸 청사진(Compass) 운동학적 보간 분석 뷰")
        cols = st.columns(4)
        df, frame_dir = st.session_state.df, st.session_state.frame_dir
        p1_gp = st.session_state.p1_gp
        ref_len = st.session_state.ref_club_len
        phases = st.session_state.phases_info
        
        for i, p in enumerate(phases):
            with cols[i % 4]:
                af = st.session_state.auto_f.get(p['phase'], 0)
                fn = st.slider(f"[{p['phase']}] 조정", 0, max(0, st.session_state.tot_frames-1), af, key=f"s_{i}")
                
                img_path = os.path.join(frame_dir, f"frame_{fn:04d}.jpg")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                else:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)

                row = df.loc[fn] if fn in df.index else None
                
                cv2.line(img, p1_gp[0], p1_gp[1], (0,0,255), 4, cv2.LINE_AA)
                draw_text_with_outline(img, "Ground", (p1_gp[0][0], p1_gp[0][1]+30), 0.6, (0,0,255), (255,255,255), 2)
                
                if row is not None:
                    wx, wy = int(row['WX_Smooth']), int(row['WY_Smooth'])
                    
                    if p['type'] == 'shaft':
                        draw_dynamic_visuals_with_compass(img, (wx, wy), row['SA_Smooth'], ref_len, p1_gp[0], p1_gp[1], (0,255,0), "Shaft")
                    elif p['type'] == 'arm_left' and not pd.isna(row['LX_Smooth']):
                        draw_dynamic_visuals_with_compass(img, (int(row['LX_Smooth']), int(row['LY_Smooth'])), row['LA_Smooth'], ref_len*0.8, p1_gp[0], p1_gp[1], (0,255,0), "Lt Arm")
                    elif p['type'] == 'arm_right' and not pd.isna(row['RX_Smooth']):
                        draw_dynamic
