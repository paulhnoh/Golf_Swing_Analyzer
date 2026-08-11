import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import os
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="P1-P13 DB Sequential Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 DB 전수 추적 분석 시스템")
st.markdown("대표님의 아이디어(DB 순차적 각도 추적)를 100% 반영한 무결점 상용화 버전입니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

# 대표님이 설계하신 시퀀셜 타겟 
phases_info = [
    {"phase": "P1", "name": "Address", "target": 90.0, "type": "shaft"},
    {"phase": "P2", "name": "Start Sweep", "target": 45.0, "type": "shaft"},
    {"phase": "P3", "name": "Back Alignment", "target": 0.0, "type": "shaft"},
    {"phase": "P4", "name": "Start Shoulder Back", "target": 0.0, "type": "arm_left"},
    {"phase": "P5", "name": "Backswing Top", "target": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "target": 315.0, "type": "shaft"},
    {"phase": "P7", "name": "DB Alignment", "target": 0.0, "type": "shaft"},
    {"phase": "P8", "name": "Impact", "target": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "target": 135.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "target": 180.0, "type": "shaft"}, # 좌측 수평(180도)
    {"phase": "P11", "name": "Start Shoulder Forward", "target": 180.0, "type": "arm_right"},
    {"phase": "P12", "name": "Downswing Top", "target": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "target": None, "type": "finish"},
]

def compute_angle(p1, p2, gp1, gp2):
    if pd.isna(p1[0]) or pd.isna(p2[0]): return np.nan
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    g_angle = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    t_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return round(math.degrees(t_angle - g_angle) % 360, 1)

def find_closest_frame(df, col, target, start_f, end_f):
    if start_f >= end_f: return start_f
    sub = df[(df['Frame'] >= start_f) & (df['Frame'] <= end_f)].copy()
    if sub.empty: return start_f
    # 360도 순환 오차 계산
    sub['diff'] = sub[col].apply(lambda x: min(abs(x - target) % 360, 360 - (abs(x - target) % 360)))
    return int(sub['diff'].idxmin())

def draw_visuals(img, vertex, target_pt, angle, gp1, gp2, color, label):
    if pd.isna(angle): return
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    g_rad = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    
    # 0도 기준선
    r_x, r_y = int(vertex[0] + 80 * math.cos(g_rad)), int(vertex[1] + 80 * math.sin(g_rad))
    cv2.line(img, vertex, (r_x, r_y), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "0", (r_x+5, r_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, "0", (r_x+5, r_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    
    cv2.circle(img, vertex, 6, (0, 255, 255), -1)
    cv2.circle(img, target_pt, 6, (0, 0, 255), -1)
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    
    # 아크 그리기
    pts = []
    for i in range(max(5, int(angle / 4)) + 1):
        a_rad = math.radians(i * (angle / max(5, int(angle / 4)))) + g_rad
        pts.append([int(vertex[0] + 45 * math.cos(a_rad)), int(vertex[1] + 45 * math.sin(a_rad))])
    if pts: cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
        
    m_rad = math.radians(angle / 2.0) + g_rad
    t_x, t_y = int(vertex[0] + 65 * math.cos(m_rad)), int(vertex[1] + 65 * math.sin(m_rad))
    text = f"{label}: {angle}deg"
    cv2.putText(img, text, (t_x-30, t_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (t_x-30, t_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

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
        
        with st.spinner("1단계: P1 지면/어드레스 캘리브레이션..."):
            ret, f_frame = cap.read()
            p1_gp = ((int(f_frame.shape[1]*0.35), int(f_frame.shape[0]*0.85)), 
                     (int(f_frame.shape[1]*0.65), int(f_frame.shape[0]*0.85)))
            p_res = pose_model(f_frame, verbose=False)[0]
            if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                kp = p_res.keypoints.xy[0].cpu().numpy()
                if len(kp) > 16 and kp[15][0] > 0 and kp[16][0] > 0:
                    p1_gp = ((int(kp[15][0]), int(kp[15][1])), (int(kp[16][0]), int(kp[16][1])))
            st.session_state.p1_gp = p1_gp

        with st.spinner("2단계: 전 프레임 추출 및 DB화 (대표님 로직 적용 중)..."):
            db_data = []
            p1_target = None
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
                        if kp[5][0] > 0: row['LX'], row['LY'] = kp[5]
                        if kp[6][0] > 0: row['RX'], row['RY'] = kp[6]
                        pts = [kp[i] for i in (9,10) if kp[i][0] > 0 and cf[i] > 0.1]
                        if pts:
                            row['WX'], row['WY'] = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
                            
                            # 💡 [Tee-Trap Filter] 클럽 헤드 오인식 원천 차단
                            v_targets = []
                            for box in c_res.boxes:
                                if float(box.conf[0]) < 0.3: continue
                                cx, cy = (box.xyxy[0][0]+box.xyxy[0][2])/2, (box.xyxy[0][1]+box.xyxy[0][3])/2
                                dist = math.hypot(cx - row['WX'], cy - row['WY'])
                                
                                # 손목이 지면에서 멀리 떨어져 있는데(스윙 중), 타겟이 공(P1) 위치에 있다면 가짜(Tee)로 간주!
                                is_tee = False
                                if p1_target and fn > 10 and row['WY'] < p1_gp[0][1] - 100:
                                    if math.hypot(cx - p1_target[0], cy - p1_target[1]) < 40:
                                        is_tee = True
                                        
                                if not is_tee and dist > 20: 
                                    v_targets.append((cx, cy, dist))
                                    
                            if v_targets:
                                best_t = max(v_targets, key=lambda x: x[2])[0]
                                row['TX'], row['TY'] = best_t
                                if fn < 5 and p1_target is None: p1_target = best_t
                
                # 순수 원시 각도 계산
                row['SA'] = compute_angle((row['WX'], row['WY']), (row['TX'], row['TY']), p1_gp[0], p1_gp[1])
                row['LA'] = compute_angle((row['LX'], row['LY']), (row['WX'], row['WY']), p1_gp[0], p1_gp[1])
                row['RA'] = compute_angle((row['RX'], row['RY']), (row['WX'], row['WY']), p1_gp[0], p1_gp[1])
                db_data.append(row)
            cap.release()
            
            # 💡 [핵심] 보간 및 벡터(Sin/Cos) 스무딩 처리 (각도 튀는 현상 제거)
            df = pd.DataFrame(db_data).interpolate(limit_direction='both')
            df['WY_Smooth'] = df['WY'].rolling(5, center=True).mean()
            
            df['SA_Sin'], df['SA_Cos'] = np.sin(np.radians(df['SA'])), np.cos(np.radians(df['SA']))
            df['SA_Smooth'] = np.degrees(np.arctan2(df['SA_Sin'].rolling(5, center=True).mean(), df['SA_Cos'].rolling(5, center=True).mean())) % 360
            
            st.session_state.df = df
            st.session_state.frame_dir = frame_dir
            st.session_state.tot_frames = tot_frames

        with st.spinner("3단계: DB 시퀀셜 역추적 (대표님 제안 로직)..."):
            # 뼈대 (손목 높낮이 기준)
            p8 = int(df.loc[int(tot_frames*0.1) : int(tot_frames*0.9)]['WY_Smooth'].idxmax()) # Impact
            p5 = int(df.loc[:p8]['WY_Smooth'].idxmin()) # Top
            p12 = int(df.loc[p8:]['WY_Smooth'].idxmin()) if p8 < tot_frames-1 else tot_frames-1 # Finish
            
            # DB 각도 추적 추출
            auto_f = {"P1": 0, "P5": p5, "P8": p8, "P12": p12, "P13": tot_frames - 1}
            auto_f["P2"] = find_closest_frame(df, 'SA_Smooth', 45.0, 0, p5)
            auto_f["P3"] = find_closest_frame(df, 'SA_Smooth', 0.0, auto_f["P2"], p5)
            auto_f["P4"] = find_closest_frame(df, 'LA', 0.0, auto_f["P3"], p5)
            auto_f["P6"] = find_closest_frame(df, 'SA_Smooth', 315.0, p5, p8)
            auto_f["P7"] = find_closest_frame(df, 'SA_Smooth', 0.0, auto_f["P6"], p8)
            auto_f["P9"] = find_closest_frame(df, 'SA_Smooth', 135.0, p8, p12)
            auto_f["P10"] = find_closest_frame(df, 'SA_Smooth', 180.0, auto_f["P9"], p12)
            auto_f["P11"] = find_closest_frame(df, 'RA', 180.0, auto_f["P10"], p12)

            st.session_state.auto_f = auto_f
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 DB 추적 기반 100% 동기화 뷰")
        cols = st.columns(4)
        df, frame_dir = st.session_state.df, st.session_state.frame_dir
        p1_gp = st.session_state.p1_gp
        
        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                af = st.session_state.auto_f.get(p['phase'], 0)
                fn = st.slider(f"[{p['phase']}] 조정", 0, st.session_state.tot_frames-1, af, key=f"s_{i}")
                
                img = cv2.imread(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"))
                row = df.loc[fn]
                
                cv2.line(img, p1_gp[0], p1_gp[1], (0,0,255), 4, cv2.LINE_AA)
                draw_text_with_outline(img, "Ground", (p1_gp[0][0], p1_gp[0][1]+30), 0.6, (0,0,255), (255,255,255), 2)
                
                wx, wy = int(row['WX']), int(row['WY'])
                
                if p['type'] == 'shaft' and not pd.isna(row['TX']):
                    draw_visuals(img, (wx, wy), (int(row['TX']), int(row['TY'])), round(row['SA_Smooth'], 1), p1_gp[0], p1_gp[1], (0,255,0), "Shaft")
                elif p['type'] == 'arm_left' and not pd.isna(row['LX']):
                    draw_visuals(img, (int(row['LX']), int(row['LY'])), (wx, wy), round(row['LA'], 1), p1_gp[0], p1_gp[1], (0,255,0), "Lt Arm")
                elif p['type'] == 'arm_right' and not pd.isna(row['RX']):
                    draw_visuals(img, (int(row['RX']), int(row['RY'])), (wx, wy), round(row['RA'], 1), p1_gp[0], p1_gp[1], (0,255,0), "Rt Arm")

                status = "Pass"
                if p['target'] is not None and not pd.isna(row['SA_Smooth']):
                    val = row['SA_Smooth'] if p['type'] == 'shaft' else (row['LA'] if p['type'] == 'arm_left' else row['RA'])
                    if min(abs(val - p['target'])%360, 360-(abs(val - p['target'])%360)) > 7.0: status = "Check"
                        
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{p['phase']}] {p['name']} ({status})", use_column_width=True)
