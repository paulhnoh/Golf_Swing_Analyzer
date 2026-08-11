"""
================================================================================
[상용화 레벨: P1-P13 청사진 100% 동기화 & 티-트랩(Tee-Trap) 마스터 엔진]
1. Tee-Trap Filter: 스윙 중 AI가 바닥의 공(Ball/Tee)을 클럽 헤드로 오인하여 
   각도가 90도 근처에 갇히는(Lock-on) 치명적 오류 완벽 차단.
2. Furthest Point Lock: 샤프트 중간이 아닌 클럽 끝단(헤드)을 정확히 잡기 위해
   손목에서 가장 먼 거리에 있는 객체를 우선순위로 추적.
3. Blueprint Math 100%: 대표님의 각도 청사진(좌=0, 하=90, 우=180, 상=270) 수학 적용.
4. Auto-Detect Handedness: 백스윙 방향에 따른 좌타/우타 자동 감지 및 타겟 반전.
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

st.set_page_config(page_title="P1-P13 Master Auto-Detect Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 청사진 100% 동기화 정밀 분석 시스템")
st.markdown("바닥의 공(Tee)을 클럽으로 오인하는 오류를 원천 차단하여 청사진 궤적을 100% 추적합니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

def get_blueprint_angle(x1, y1, x2, y2, gp1, gp2):
    """💡 [핵심] 대표님의 청사진(Left=0, Down=90, Right=180, Up=270) 수학 공식 100% 적용"""
    if pd.isna(x1) or pd.isna(x2): return np.nan
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    
    g_angle = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    dx = x2 - x1
    dy = y2 - y1
    
    # 지면 기울기 보정 및 좌/우 반전(청사진 동기화)
    dx_rot = dx * math.cos(-g_angle) - dy * math.sin(-g_angle)
    dy_rot = dx * math.sin(-g_angle) + dy * math.cos(-g_angle)
    
    t_angle = math.atan2(dy_rot, -dx_rot) # -dx_rot을 통해 좌측을 0(360)으로 강제
    val = math.degrees(t_angle)
    if val < 0: val += 360
    return round(val, 1)

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

def draw_dynamic_visuals_with_compass(img, vertex, angle, length, gp1, gp2, color, label):
    """청사진 나침반 배경 및 동적 오버레이 렌더링"""
    if pd.isna(angle) or pd.isna(vertex[0]): return
    if gp1[0] > gp2[0]: gp1, gp2 = gp2, gp1
    g_angle = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    
    # 1. 청사진 방위각(Compass) 8방향 렌더링
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

    # 2. 타겟 각도 렌더링
    t_rad = math.radians(angle)
    tx = -math.cos(t_rad) * math.cos(g_angle) - math.sin(t_rad) * math.sin(g_angle)
    ty = -math.cos(t_rad) * math.sin(g_angle) + math.sin(t_rad) * math.cos(g_angle)
    target_pt = (int(vertex[0] + length * tx), int(vertex[1] + length * ty))
    
    cv2.circle(img, vertex, 6, (0, 255, 255), -1)
    cv2.circle(img, target_pt, 6, (0, 0, 255), -1)
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    
    # 3. 아크(Arc) 렌더링
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
        
        with st.spinner("1단계: 기초 캘리브레이션 중..."):
            ret, f_frame = cap.read()
            p1_gp = ((int(f_frame.shape[1]*0.35), int(f_frame.shape[0]*0.85)), 
                     (int(f_frame.shape[1]*0.65), int(f_frame.shape[0]*0.85)))
            
            p_res = pose_model(f_frame, verbose=False)[0]
            if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                kp = p_res.keypoints.xy[0].cpu().numpy()
                if len(kp) > 16 and kp[15][0] > 0 and kp[16][0] > 0:
                    p1_gp = ((int(kp[15][0]), int(kp[15][1])), (int(kp[16][0]), int(kp[16][1])))
            st.session_state.p1_gp = p1_gp
            st.session_state.ref_club_len = f_frame.shape[1] * 0.3

            db_data = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            wx_start, wy_start = None, None
            p1_target = None # 어드레스 시점의 타겟(골프공) 위치 저장

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
                            if wx_start is None: wx_start, wy_start = row['WX'], row['WY']
                            
                            # 💡 [핵심 방어] 티-트랩(Tee-Trap) 필터 및 최장거리(Furthest Point) 우선순위
                            v_targets = []
                            for box in c_res.boxes:
                                conf = float(box.conf[0].item())
                                if conf < 0.2: continue
                                
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cx, cy = float((x1+x2)/2.0), float((y1+y2)/2.0)
                                dist = math.hypot(cx - row['WX'], cy - row['WY'])
                                
                                # 거리 너무 멀면 아웃 (엉뚱한 사람 등)
                                if dist > st.session_state.ref_club_len * 2.0: continue
                                
                                # 💡 공(Ball/Tee)에 록인되는 치명적 버그 완전 봉쇄!
                                if p1_target is not None:
                                    hands_moved = math.hypot(row['WX'] - wx_start, row['WY'] - wy_start) > 30
                                    # 손목이 움직였는데 타겟이 공(시작 위치) 근처라면 가짜로 판별!
                                    if hands_moved and math.hypot(cx - p1_target[0], cy - p1_target[1]) < 40:
                                        continue 
                                        
                                v_targets.append((cx, cy, dist, conf))
                                    
                            if v_targets:
                                # 💡 샤프트 중심이 아닌 '클럽 끝단'을 잡기 위해 거리가 가장 먼 놈(x[2])을 선택
                                best_t = max(v_targets, key=lambda x: x[2]) 
                                row['TX'], row['TY'] = best_t[0], best_t[1]
                                
                                # 최초의 타겟(공/티) 좌표 기억
                                if fn < 5 and p1_target is None:
                                    p1_target = (best_t[0], best_t[1])
                                    st.session_state.ref_club_len = best_t[2] if best_t[2] > 50 else st.session_state.ref_club_len
                                
                db_data.append(row)
            cap.release()

        with st.spinner("2단계: 데이터 스무딩 및 완벽한 수학적 궤적 산출 중..."):
            df = pd.DataFrame(db_data)
            df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']] = df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']].interpolate(limit_direction='both')
            
            for col in ['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']:
                df[f'{col}_Smooth'] = df[col].rolling(window=5, min_periods=1, center=True).mean()

            # 스무딩된 절대 좌표를 바탕으로 튀지 않는 완벽한 청사진 각도 계산
            for i in df.index:
                df.loc[i, 'SA_Smooth'] = get_blueprint_angle(df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], df.loc[i, 'TX_Smooth'], df.loc[i, 'TY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'LA_Smooth'] = get_blueprint_angle(df.loc[i, 'LX_Smooth'], df.loc[i, 'LY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'RA_Smooth'] = get_blueprint_angle(df.loc[i, 'RX_Smooth'], df.loc[i, 'RY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])

        with st.spinner("3단계: 좌/우타 자동 감지 및 시퀀스 타임라인 구축 중..."):
            # V-Curve 뼈대 구축
            p5 = int(df['WY_Smooth'].iloc[:int(tot_frames * 0.65)].idxmin())
            p12 = int(df['WY_Smooth'].iloc[p5 + 15 :].idxmin()) if len(df.iloc[p5 + 15 :]) > 0 else tot_frames - 1
            sub_imp = df['WY_Smooth'].iloc[p5 + 5 : p12 - 5]
            p8 = int(sub_imp.idxmax()) if not sub_imp.empty else p5 + (p12 - p5) // 2
            
            wx_start_avg = df['WX_Smooth'].iloc[0:5].mean()
            p1_mask = (df['WX_Smooth'].iloc[:p5] - wx_start_avg).abs() > 3.0
            p1 = int(p1_mask.idxmax()) if p1_mask.any() else 0

            # 좌타자/우타자 0.1초 자동 판별
            is_left_handed = df['WX_Smooth'].iloc[p5] < df['WX_Smooth'].iloc[p1]

            if is_left_handed:
                st.toast("🏌️‍♂️ 좌타자(Left-handed) 스윙 궤적이 감지되었습니다. 타겟 각도를 반전합니다!", icon="🔄")
                tgt_p2, tgt_p3, tgt_p6, tgt_p7, tgt_p9, tgt_p10 = 45.0, 0.0, 315.0, 0.0, 180.0, 180.0
                tgt_p4_arm, tgt_p4_ang = 'RA_Smooth', 0.0
                tgt_p11_arm, tgt_p11_ang = 'LA_Smooth', 180.0
            else:
                st.toast("🏌️‍♂️ 우타자(Right-handed) 스윙 궤적이 감지되었습니다.", icon="✅")
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
        st.subheader("📸 청사진(Compass) 좌/우타 100% 동기화 분석 뷰")
        cols = st.columns(4)
        df, frame_dir = st.session_state.df, st.session_state.frame_dir
        p1_gp = st.session_state.p1_gp
        ref_len = st.session_state.ref_club_len
        phases = st.session_state.phases_info
        
        for i, p in enumerate(phases):
            with cols[i % 4]:
                af = st.session_state.auto_f.get(p['phase'], 0)
                fn = st.slider(f"[{p['phase']}] 조정", 0, st.session_state.tot_frames-1, af, key=f"s_{i}")
                
                img = cv2.imread(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"))
                row = df.loc[fn]
                
                cv2.line(img, p1_gp[0], p1_gp[1], (0,0,255), 4, cv2.LINE_AA)
                draw_text_with_outline(img, "Ground", (p1_gp[0][0], p1_gp[0][1]+30), 0.6, (0,0,255), (255,255,255), 2)
                
                wx, wy = int(row['WX_Smooth']), int(row['WY_Smooth'])
                
                if p['type'] == 'shaft':
                    draw_dynamic_visuals_with_compass(img, (wx, wy), row['SA_Smooth'], ref_len, p1_gp[0], p1_gp[1], (0,255,0), "Shaft")
                elif p['type'] == 'arm_left' and not pd.isna(row['LX_Smooth']):
                    draw_dynamic_visuals_with_compass(img, (int(row['LX_Smooth']), int(row['LY_Smooth'])), row['LA_Smooth'], ref_len*0.8, p1_gp[0], p1_gp[1], (0,255,0), "Lt Arm")
                elif p['type'] == 'arm_right' and not pd.isna(row['RX_Smooth']):
                    draw_dynamic_visuals_with_compass(img, (int(row['RX_Smooth']), int(row['RY_Smooth'])), row['RA_Smooth'], ref_len*0.8, p1_gp[0], p1_gp[1], (0,255,0), "Rt Arm")

                status = "Pass"
                if p['target'] is not None:
                    val = row['SA_Smooth'] if p['type'] == 'shaft' else (row['LA_Smooth'] if p['type'] == 'arm_left' else row['RA_Smooth'])
                    if not pd.isna(val) and min(abs(val - p['target'])%360, 360-(abs(val - p['target'])%360)) > 7.0: 
                        status = "Check"
                        
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{p['phase']}] {p['name']} ({status})", use_column_width=True)
