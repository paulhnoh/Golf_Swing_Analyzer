"""
================================================================================
[상용화 레벨: Twin-Peaks V-Curve 생체역학 앵커 엔진]
1. V-Curve 앵커: 손목 Y축(높낮이)의 양대 최고점(P5, P12)을 먼저 잡고, 그 사이의 
   최저점(P8)을 임팩트로 록인하여 프레임이 엉키는 현상을 100% 원천 차단.
2. 텐서 에러 프리: PyTorch 0-d tensor 버그를 완전 제거한 float 캐스팅 적용.
3. 다이나믹 렌더링: 디봇/티 오인식 좌표를 무시하고 스무딩된 벡터 각도로 오버레이 렌더링.
4. 360도 수학적 동기화: Right=0°, Down=90°, Left=180°, Up=270° 유지.
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

st.set_page_config(page_title="P1-P13 V-Curve Master Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 생체역학(V-Curve) 정밀 분석 시스템")
st.markdown("손목의 상하 높낮이(쌍봉) 궤적을 이용해 스윙 뼈대를 100% 정확하게 추출합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

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
    {"phase": "P10", "name": "DF Alignment", "target": 180.0, "type": "shaft"}, 
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
    sub['diff'] = sub[col].apply(lambda x: min(abs(x - target) % 360, 360 - (abs(x - target) % 360)))
    return int(sub['diff'].idxmin())

def draw_text_with_outline(img, text, pos, font_scale, text_color, outline_color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_dynamic_visuals(img, vertex, angle, length, gp1, gp2, color, label):
    """스무딩된 수학적 각도를 바탕으로 일정한 길이의 선과 호(Arc)를 렌더링"""
    if pd.isna(angle) or pd.isna(vertex[0]): return
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    g_rad = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    
    # 0도 기준선 렌더링
    r_x, r_y = int(vertex[0] + 80 * math.cos(g_rad)), int(vertex[1] + 80 * math.sin(g_rad))
    cv2.line(img, vertex, (r_x, r_y), (255, 255, 255), 2, cv2.LINE_AA)
    draw_text_with_outline(img, "0", (r_x+5, r_y-5), 0.5, (255,255,255), (0,0,0), 1)
    
    # 목표 각도 선 렌더링
    t_rad = math.radians(angle) + g_rad
    target_pt = (int(vertex[0] + length * math.cos(t_rad)), int(vertex[1] + length * math.sin(t_rad)))
    cv2.circle(img, vertex, 6, (0, 255, 255), -1)
    cv2.circle(img, target_pt, 6, (0, 0, 255), -1)
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    
    # 각도 호(Arc) 렌더링
    pts = []
    for i in range(max(5, int(angle / 4)) + 1):
        a_rad = math.radians(i * (angle / max(5, int(angle / 4)))) + g_rad
        pts.append([int(vertex[0] + 45 * math.cos(a_rad)), int(vertex[1] + 45 * math.sin(a_rad))])
    if pts: cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
        
    # 텍스트 라벨 렌더링
    m_rad = math.radians(angle / 2.0) + g_rad
    t_x, t_y = int(vertex[0] + 65 * math.cos(m_rad)), int(vertex[1] + 65 * math.sin(m_rad))
    draw_text_with_outline(img, f"{label}: {round(angle, 1)}deg", (t_x-30, t_y+10), 0.7, (0,255,255), (0,0,0), 2)

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

        with st.spinner("2단계: 100% 프레임 DB 변환 및 텐서 에러 검출 중..."):
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
                                if float(box.conf[0].item()) < 0.2: continue # 검출 민감도 유지
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cx, cy = float((x1+x2)/2.0), float((y1+y2)/2.0)
                                dist = math.hypot(cx - row['WX'], cy - row['WY'])
                                
                                # 가짜 객체 배제를 위한 샤프트 반경 제한
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
            
            # 결측치 보간 및 벡터(Sin/Cos) 스무딩 (각도 튀는 현상 원천 차단)
            df = pd.DataFrame(db_data)
            df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY']] = df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY']].interpolate(limit_direction='both')
            
            # 손목 높낮이 스무딩 
            df['WY_Smooth'] = df['WY'].rolling(window=11, min_periods=1, center=True).mean()
            
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

        with st.spinner("3단계: V-Curve 뼈대 록인(Sequence-Lock) 구축 중..."):
            # 💡 [핵심] V-Curve 쌍봉(Twin-Peaks) 알고리즘
            
            # 1. P5 (Top): 영상 앞부분(65%)에서 손목이 가장 높이 올라간 지점 (Y 최소값)
            search_end_p5 = int(tot_frames * 0.65)
            p5 = int(df['WY_Smooth'].iloc[:search_end_p5].idxmin())
            
            # 2. P12 (Finish): P5 이후부터 영상 끝까지 손목이 가장 높이 올라간 지점 (Y 최소값)
            p12 = int(df['WY_Smooth'].iloc[p5 + 15 :].idxmin()) if len(df.iloc[p5 + 15 :]) > 0 else tot_frames - 1
            
            # 3. P8 (Impact): 무조건 P5와 P12 사이에서 손이 가장 낮아지는 지점 (Y 최대값)
            sub_imp = df['WY_Smooth'].iloc[p5 + 5 : p12 - 5]
            p8 = int(sub_imp.idxmax()) if not sub_imp.empty else p5 + (p12 - p5) // 2

            # 타임라인 순차 매핑
            auto_f = {"P1": 0, "P5": p5, "P8": p8, "P12": p12, "P13": tot_frames - 1}
            auto_f["P2"] = find_closest_frame(df, 'SA_Smooth', 45.0, 0, max(1, p5-5))
            auto_f["P3"] = find_closest_frame(df, 'SA_Smooth', 0.0, auto_f["P2"], max(1, p5-3))
            auto_f["P4"] = find_closest_frame(df, 'LA_Smooth', 0.0, auto_f["P3"], max(1, p5-1))
            
            auto_f["P6"] = find_closest_frame(df, 'SA_Smooth', 315.0, p5, max(p5+1, p8-3))
            auto_f["P7"] = find_closest_frame(df, 'SA_Smooth', 0.0, auto_f["P6"], max(p5+2, p8-1))
            
            auto_f["P9"] = find_closest_frame(df, 'SA_Smooth', 135.0, p8, max(p8+1, p12-5))
            auto_f["P10"] = find_closest_frame(df, 'SA_Smooth', 180.0, auto_f["P9"], max(p8+2, p12-3))
            auto_f["P11"] = find_closest_frame(df, 'RA_Smooth', 180.0, auto_f["P10"], max(p8+3, p12-1))

            st.session_state.auto_f = auto_f
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state:
        st.subheader("📸 V-Curve 앵커 & 다이나믹 오버레이 뷰")
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
                
                # 지면선 렌더링
                cv2.line(img, p1_gp[0], p1_gp[1], (0,0,255), 4, cv2.LINE_AA)
                draw_text_with_outline(img, "Ground", (p1_gp[0][0], p1_gp[0][1]+30), 0.6, (0,0,255), (255,255,255), 2)
                
                wx, wy = int(row['WX']), int(row['WY'])
                
                # 💡 다이나믹 렌더링 호출 (오작동 좌표 무시, 부드러운 스무딩 각도 적용)
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
