"""
================================================================================
[상용화 레벨: P1-P13 무결점 통합 마스터 엔진 (Blueprint Angle Fix)]
1. 각도 청사진 100% 적용: Left=0(360°), Down=90°, Right=180°, Up=270° 체계 절대 확립.
2. P1 모션 감지 (Start Sweep): 손목(Wrist X)이 최초로 움직이는 프레임을 P1으로 자동 감지.
3. V-Curve 앵커 & 시퀀스 록인: P5(탑), P8(임팩트), P12(피니시) 상하 뼈대 절대 보장.
4. 다이나믹 렌더링 교정: 0도 기준선이 좌측을 향하고 청사진에 맞춘 완벽한 Arc 시각화.
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

st.set_page_config(page_title="P1-P13 Precision Blueprint Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 각도 청사진 정밀 분석 시스템")
st.markdown("대표님의 '좌측=0도, 우측=180도' 청사진이 100% 이식된 최종 상용화 버전입니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

# 💡 [핵심] 대표님의 원래 설계대로 타겟 각도 원상 복구!
phases_info = [
    {"phase": "P1", "name": "Address", "target": None, "type": "shaft"}, # 모션 시작점 자동 감지
    {"phase": "P2", "name": "Start Sweep", "target": 45.0, "type": "shaft"}, # 0도(좌)에서 45도(하단) 방향
    {"phase": "P3", "name": "Back Alignment", "target": 0.0, "type": "shaft"}, # 좌측 수평
    {"phase": "P4", "name": "Start Shoulder Back", "target": 0.0, "type": "arm_left"}, # 좌측 수평
    {"phase": "P5", "name": "Backswing Top", "target": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "target": 315.0, "type": "shaft"}, # 좌상단 방향
    {"phase": "P7", "name": "DB Alignment", "target": 0.0, "type": "shaft"}, # 좌측 수평
    {"phase": "P8", "name": "Impact", "target": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "target": 135.0, "type": "shaft"}, # 우하단 방향
    {"phase": "P10", "name": "DF Alignment", "target": 180.0, "type": "shaft"}, # 우측 수평
    {"phase": "P11", "name": "Start Shoulder Forward", "target": 180.0, "type": "arm_right"}, # 우측 수평
    {"phase": "P12", "name": "Downswing Top", "target": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "target": None, "type": "finish"},
]

def compute_angle(p1, p2, gp1, gp2):
    """💡 [핵심] 청사진 이식: Left=0, Down=90, Right=180, Up=270 변환 공식"""
    if pd.isna(p1[0]) or pd.isna(p2[0]): return np.nan
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    g_angle = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    
    t_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    std_angle = math.degrees(t_angle - g_angle)
    
    # 표준 각도를 대표님의 청사진 각도로 변환 (180도를 빼고 뒤집음)
    blueprint_angle = (180 - std_angle) % 360
    return round(blueprint_angle, 1)

def find_closest_frame(df, col, target, start_f, end_f):
    if start_f >= end_f: return start_f
    sub = df[(df['Frame'] >= start_f) & (df['Frame'] <= end_f)].copy()
    if sub.empty: return start_f
    sub['diff'] = sub[col].apply(lambda x: min(abs(x - target) % 360, 360 - (abs(x - target) % 360)))
    return int(sub['diff'].idxmin())

def draw_text_with_outline(img, text, pos, font_scale, text_color, outline_color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_dynamic_visuals(img, vertex, blueprint_angle, length, gp1, gp2, color, label):
    """💡 [핵심] 청사진 시각화: 0도 기준선이 화면 좌측으로 뻗어나감"""
    if pd.isna(blueprint_angle) or pd.isna(vertex[0]): return
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    g_rad = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    
    # 청사진의 0도(좌측) 기준선 렌더링
    r_rad = math.radians(180) + g_rad 
    r_x, r_y = int(vertex[0] + 80 * math.cos(r_rad)), int(vertex[1] + 80 * math.sin(r_rad))
    cv2.line(img, vertex, (r_x, r_y), (255, 255, 255), 2, cv2.LINE_AA)
    draw_text_with_outline(img, "0(360)", (r_x-60, r_y-5), 0.5, (255,255,255), (0,0,0), 1)
    
    # 목표 각도(청사진) -> 렌더링을 위한 원상 복구 라디안 계산
    std_angle = (180 - blueprint_angle) % 360
    t_rad = math.radians(std_angle) + g_rad
    target_pt = (int(vertex[0] + length * math.cos(t_rad)), int(vertex[1] + length * math.sin(t_rad)))
    
    cv2.circle(img, vertex, 6, (0, 255, 255), -1)
    cv2.circle(img, target_pt, 6, (0, 0, 255), -1)
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    
    # 청사진 기준 0도(좌측)부터 시작하는 호(Arc) 그리기
    pts = []
    num_steps = max(5, int(blueprint_angle / 4))
    for i in range(num_steps + 1):
        curr_b_angle = i * (blueprint_angle / num_steps) if num_steps > 0 else 0
        curr_std_angle = (180 - curr_b_angle) % 360
        a_rad = math.radians(curr_std_angle) + g_rad
        pts.append([int(vertex[0] + 45 * math.cos(a_rad)), int(vertex[1] + 45 * math.sin(a_rad))])
    
    if pts: cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
        
    mid_b_angle = blueprint_angle / 2.0
    mid_std_angle = (180 - mid_b_angle) % 360
    m_rad = math.radians(mid_std_angle) + g_rad
    t_x, t_y = int(vertex[0] + 65 * math.cos(m_rad)), int(vertex[1] + 65 * math.sin(m_rad))
    draw_text_with_outline(img, f"{label}: {round(blueprint_angle, 1)}deg", (t_x-40, t_y+15), 0.7, (0,255,255), (0,0,0), 2)

uploaded_file = st.file_uploader("스윙 영상 업로드 (MP4, MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'curr_file' not in st.session_state or st.session_state.curr_file != uploaded_file.name:
        st.session_state.clear()
        st.session_state.curr_file = uploaded_file.name

    if 'scan_done' not in st.session_state:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        frame_dir = tempfile.mkdtemp()
        
        cap = cv2.VideoCapture(tfile.name)
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with st.spinner("1단계: P1 지면/어드레스 캘리브레이션 중..."):
            ret, f_frame = cap.read()
            p1_gp = ((int(f_frame.shape[1]*0.35), int(f_frame.shape[0]*0.85)), 
                     (int(f_frame.shape[1]*0.65), int(f_frame.shape[0]*0.85)))
            
            p_res = pose_model(f_frame, verbose=False)[0]
            if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                kp = p_res.keypoints.xy[0].cpu().numpy()
                if len(kp) > 16 and kp[15][0] > 0 and kp[16][0] > 0:
                    p1_gp = ((int(kp[15][0]), int(kp[15][1])), (int(kp[16][0]), int(kp[16][1])))
            st.session_state.p1_gp = p1_gp
            
            ref_club_len = f_frame.shape[1] * 0.3
            c_res = custom_model(f_frame, verbose=False)[0]
            if len(kp) > 10 and kp[9][0] > 0 and kp[10][0] > 0:
                wpt = (int((kp[9][0]+kp[10][0])/2), int((kp[9][1]+kp[10][1])/2))
                for box in c_res.boxes:
                    if c_res.names[int(box.cls[0])] in ['head', 'shaft']:
                        cx, cy = (box.xyxy[0][0]+box.xyxy[0][2])/2, (box.xyxy[0][1]+box.xyxy[0][3])/2
                        ref_club_len = float(math.hypot(cx - wpt[0], cy - wpt[1]))
                        break
            st.session_state.ref_club_len = ref_club_len

        with st.spinner("2단계: 전 프레임 추출 및 데이터베이스(DB) 변환 중..."):
            db_data = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            for fn in range(tot_frames):
                ret, frame = cap.read()
                if not ret: break
                cv2.imwrite(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"), frame)
                
                p_res = pose_model(frame, verbose=False)[0]
                c_res = custom_model(frame, verbose=False)[0]
                
                row = {'Frame': fn, 'WX': np.nan, 'WY': np.nan, 'TX': np.nan, 'TY': np.nan,
                       'LX': np.nan, 'LY': np.nan, 'RX': np.nan, 'RY': np.nan}
                
                if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                    kp = p_res.keypoints.xy[0].cpu().numpy()
                    cf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints.conf is not None else np.ones(17)
                    
                    if len(kp) > 10:
                        if kp[5][0] > 0: row['LX'], row['LY'] = float(kp[5][0]), float(kp[5][1])
                        if kp[6][0] > 0: row['RX'], row['RY'] = float(kp[6][0]), float(kp[6][1])
                        pts = [kp[i] for i in (9,10) if kp[i][0] > 0 and cf[i] > 0.1]
                        
                        if pts:
                            row['WX'], row['WY'] = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
                            
                            v_targets = []
                            for box in c_res.boxes:
                                if float(box.conf[0].item()) < 0.2: continue 
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cx, cy = float((x1+x2)/2.0), float((y1+y2)/2.0)
                                dist = math.hypot(cx - row['WX'], cy - row['WY'])
                                if (ref_club_len * 0.5) < dist < (ref_club_len * 1.5):
                                    v_targets.append((cx, cy, dist))
                                    
                            if v_targets:
                                best_t = max(v_targets, key=lambda x: x[2])
                                row['TX'], row['TY'] = float(best_t[0]), float(best_t[1])
                                
                if not pd.isna(row['TX']):
                    row['SA'] = compute_angle((row['WX'], row['WY']), (row['TX'], row['TY']), p1_gp[0], p1_gp[1])
                row['LA'] = compute_angle((row['LX'], row['LY']), (row['WX'], row['WY']), p1_gp[0], p1_gp[1])
                row['RA'] = compute_angle((row['RX'], row['RY']), (row['WX'], row['WY']), p1_gp[0], p1_gp[1])
                db_data.append(row)
            cap.release()
            
            # 결측치 보간 및 사인/코사인 벡터 스무딩 (부드러운 각도)
            df = pd.DataFrame(db_data)
            df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY']] = df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY']].interpolate(limit_direction='both')
            df['WX_Smooth'] = df['WX'].rolling(window=7, min_periods=1, center=True).mean()
            df['WY_Smooth'] = df['WY'].rolling(window=7, min_periods=1, center=True).mean()
            
            for col in ['SA', 'LA', 'RA']:
                df[f'{col}_Sin'] = np.sin(np.radians(df[col])).interpolate(limit_direction='both')
                df[f'{col}_Cos'] = np.cos(np.radians(df[col])).interpolate(limit_direction='both')
                df[f'{col}_Smooth'] = np.degrees(np.arctan2(
                    df[f'{col}_Sin'].rolling(7, min_periods=1, center=True).mean(), 
                    df[f'{col}_Cos'].rolling(7, min_periods=1, center=True).mean()
                )) % 360
            
            st.session_state.df = df
            st.session_state.frame_dir = frame_dir
            st.session_state.tot_frames = tot_frames

        with st.spinner("3단계: 생체역학 앵커링 & 청사진 기반 시퀀스 추적 중..."):
            # 1. P5(Top)와 P12(Finish) 절대 록인
            search_end_p5 = int(tot_frames * 0.65)
            p5 = int(df['WY_Smooth'].iloc[:search_end_p5].idxmin())
            
            p12 = int(df['WY_Smooth'].iloc[p5 + 15 :].idxmin()) if len(df.iloc[p5 + 15 :]) > 0 else tot_frames - 1
            
            # 2. P8(Impact): P5와 P12 사이에서 가장 Y가 높은(손이 내려간) 점
            sub_imp = df['WY_Smooth'].iloc[p5 + 5 : p12 - 5]
            p8 = int(sub_imp.idxmax()) if not sub_imp.empty else p5 + (p12 - p5) // 2
            
            # 3. P1(Address): 정지된 손이 '최초로 이동을 시작하는 프레임' (모션 감지 로직)
            wx_avg_start = df['WX_Smooth'].iloc[0:5].mean()
            p1_mask = (df['WX_Smooth'].iloc[:p5] - wx_avg_start).abs() > 5.0
            p1 = int(p1_mask.idxmax()) if p1_mask.any() else 0

            # 4. 좌우 교정된 시퀀스 타임라인 구축
            auto_f = {"P1": p1, "P5": p5, "P8": p8, "P12": p12, "P13": tot_frames - 1}
            auto_f["P2"] = find_closest_frame(df, 'SA_Smooth', 45.0, p1, p5)
            auto_f["P3"] = find_closest_frame(df, 'SA_Smooth', 0.0, auto_f["P2"], p5)
            auto_f["P4"] = find_closest_frame(df, 'LA_Smooth', 0.0, p1, p5)
            
            auto_f["P6"] = find_closest_frame(df, 'SA_Smooth', 315.0, p5, p8)
            auto_f["P7"] = find_closest_frame(df, 'SA_Smooth', 0.0, auto_f["P6"], p8)
            
            auto_f["P9"] = find_closest_frame(df, 'SA_Smooth', 135.0, p8, p12)
            auto_f["P10"] = find_closest_frame(df, 'SA_Smooth', 180.0, auto_f["P9"], p12)
            auto_f["P11"] = find_closest_frame(df, 'RA_Smooth', 180.0, auto_f["P10"], p12)

            st.session_state.auto_f = auto_f
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 청사진(Blueprint) 완벽 동기화 분석 뷰")
        cols = st.columns(4)
        df, frame_dir = st.session_state.df, st.session_state.frame_dir
        p1_gp = st.session_state.p1_gp
        ref_len = st.session_state.ref_club_len
        
        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                af = st.session_state.auto_f.get(p['phase'], 0)
                fn = st.slider(f"[{p['phase']}] 조정", 0, st.session_state.tot_frames-1, af, key=f"s_{i}")
                
                img = cv2.imread(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"))
                row = df.loc[fn]
                
                cv2.line(img, p1_gp[0], p1_gp[1], (0,0,255), 4, cv2.LINE_AA)
                draw_text_with_outline(img, "Ground", (p1_gp[0][0], p1_gp[0][1]+30), 0.6, (0,0,255), (255,255,255), 2)
                
                wx, wy = int(row['WX']), int(row['WY'])
                
                # 다이나믹 렌더링
                if p['type'] == 'shaft':
                    draw_dynamic_visuals(img, (wx, wy), row['SA_Smooth'], ref_len, p1_gp[0], p1_gp[1], (0,255,0), "Shaft")
                elif p['type'] == 'arm_left' and not pd.isna(row['LX']):
                    draw_dynamic_visuals(img, (int(row['LX']), int(row['LY'])), row['LA_Smooth'], ref_len*0.8, p1_gp[0], p1_gp[1], (0,255,0), "Lt Arm")
                elif p['type'] == 'arm_right' and not pd.isna(row['RX']):
                    draw_dynamic_visuals(img, (int(row['RX']), int(row['RY'])), row['RA_Smooth'], ref_len*0.8, p1_gp[0], p1_gp[1], (0,255,0), "Rt Arm")

                status = "Pass"
                if p['target'] is not None:
                    val = row['SA_Smooth'] if p['type'] == 'shaft' else (row['LA_Smooth'] if p['type'] == 'arm_left' else row['RA_Smooth'])
                    if not pd.isna(val) and min(abs(val - p['target'])%360, 360-(abs(val - p['target'])%360)) > 7.0: 
                        status = "Check"
                        
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{p['phase']}] {p['name']} ({status})", use_column_width=True)
