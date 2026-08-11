"""
================================================================================
[상용화 레벨: P1-P13 무결점 통합 마스터 엔진 (Syntax Error Fix)]
1. 괄호 누락 및 SyntaxError 원인 제거 (angle_diff 함수 도입)
2. 청사진 나침반(Compass) 100% 동기화 렌더링 유지
3. 메모리 충돌 방지 및 안전한 프레임 렌더링 보장
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

st.set_page_config(page_title="P1-P13 Master Clean Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 정밀 분석 시스템")
st.markdown("ROI 크롭 + 칼만 필터 추적으로 클럽 헤드/샤프트 인식 안정성을 강화한 버전입니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

with st.sidebar:
    st.header("🔧 클럽 탐지 튜닝")
    st.caption("샤프트/헤드를 자주 놓친다면 탐색 반경을 늘리거나 신뢰도를 낮춰보세요.")
    conf_th = st.slider("클럽 탐지 신뢰도(Confidence)", 0.05, 0.5, 0.12, 0.01,
                         help="ROI로 좁혀서 탐지하므로 전체 프레임 탐지보다 낮게 잡아도 안전합니다.")
    search_radius_ratio = st.slider("탐색 반경 비율 (× 기준 클럽 길이)", 0.2, 1.2, 0.55, 0.05,
                                     help="칼만 예측 위치를 중심으로 이 반경 안에서만 클럽을 찾습니다. 너무 좁으면 빠른 스윙 구간에서 놓칩니다.")
    det_imgsz = st.select_slider("ROI 탐지 해상도(imgsz)", options=[320, 480, 640, 800, 960], value=640,
                                  help="크롭된 영역을 이 크기로 확대해서 탐지합니다. 작은 헤드/샤프트일수록 높이는 게 유리합니다.")
    if st.button("🔄 현재 설정으로 다시 분석"):
        for k in ['scan_done', 'df', 'frame_dir', 'p1_gp', 'ref_club_len', 'auto_f', 'tot_frames', 'phases_info']:
            st.session_state.pop(k, None)
        st.rerun()

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

def angle_diff(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def create_kalman_2d(init_x, init_y):
    """등속도 모델 2D 칼만 필터. 클럽 헤드/샤프트가 잠깐 탐지되지 않아도
    직전 속도를 바탕으로 위치를 예측해 추적이 끊기지 않도록 해준다."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)
    # 골프 스윙은 가속이 크므로 프로세스 노이즈를 넉넉히 잡아 급격한 방향 전환에도 반응하게 함
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 8.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
    kf.statePost = np.array([[np.float32(init_x)], [np.float32(init_y)], [0], [0]], dtype=np.float32)
    return kf

def detect_club_in_roi(frame, model, pred_x, pred_y, roi_half, wx, wy, conf_th, imgsz):
    """예측 위치(pred_x, pred_y) 주변만 잘라서 확대 탐지 -> 작은 헤드/샤프트 인식률 개선.
    후보는 예측 위치와의 거리로 게이팅해서 배경의 엉뚱한 오탐을 걸러낸다."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(pred_x - roi_half)), max(0, int(pred_y - roi_half))
    x1, y1 = min(w, int(pred_x + roi_half)), min(h, int(pred_y + roi_half))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    crop = frame[y0:y1, x0:x1]
    try:
        c_res = model(crop, verbose=False, conf=conf_th, imgsz=imgsz)[0]
    except Exception:
        return None

    head_c, shaft_c = [], []
    if c_res.boxes is not None and len(c_res.boxes) > 0:
        for box in c_res.boxes:
            conf = float(box.conf[0].item())
            cls_idx = int(box.cls[0].item())
            cls_name = str(c_res.names.get(cls_idx, cls_idx)).lower()
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            cx, cy = x0 + float((bx1 + bx2) / 2.0), y0 + float((by1 + by2) / 2.0)
            dist_pred = math.hypot(cx - pred_x, cy - pred_y)
            if dist_pred > roi_half * 1.3:
                continue
            if 'head' in cls_name or 'club' in cls_name:
                head_c.append((cx, cy, conf, dist_pred))
            else:
                shaft_c.append((cx, cy, conf, dist_pred))

    if head_c:
        best = max(head_c, key=lambda x: x[2] - 0.15 * (x[3] / max(roi_half, 1)))
        return best[0], best[1]
    if shaft_c:
        best = max(shaft_c, key=lambda x: x[2] - 0.35 * (x[3] / max(roi_half, 1)))
        return best[0], best[1]
    return None

def find_closest_frame(df, col, target, start_f, end_f):
    if start_f >= end_f or col not in df.columns: return start_f
    sub = df[(df['Frame'] >= start_f) & (df['Frame'] <= end_f)].copy()
    if sub.empty: return start_f
    sub['diff'] = sub[col].apply(lambda x: angle_diff(x, target) if not pd.isna(x) else 999)
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
        
        with st.spinner("1단계: 기초 캘리브레이션 및 방어형 스캔 중..."):
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
            kf_club = None          # 클럽(헤드/샤프트) 위치 추적용 칼만 필터. 탐지가 끊겨도 이걸로 위치를 이어감
            roi_half = max(70, st.session_state.ref_club_len * search_radius_ratio)
            miss_streak = 0         # 연속 미탐지 횟수. 너무 오래 놓치면 탐색 반경을 넓혀 재탐색

            for fn in range(tot_frames):
                ret, frame = cap.read()
                if not ret or frame is None: break
                cv2.imwrite(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"), frame)

                row = {'Frame': fn, 'WX': np.nan, 'WY': np.nan, 'TX': np.nan, 'TY': np.nan,
                       'LX': np.nan, 'LY': np.nan, 'RX': np.nan, 'RY': np.nan, 'T_Predicted': False}

                try:
                    p_res = pose_model(frame, verbose=False)[0]

                    if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                        kp = p_res.keypoints.xy[0].cpu().numpy()
                        cf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints.conf is not None else np.ones(17)

                        if len(kp) > 10:
                            if kp[5][0] > 0: row['LX'], row['LY'] = float(kp[5][0]), float(kp[5][1])
                            if kp[6][0] > 0: row['RX'], row['RY'] = float(kp[6][0]), float(kp[6][1])
                            pts = [kp[i] for i in (9, 10) if kp[i][0] > 0 and cf[i] > 0.05]

                            if pts:
                                row['WX'], row['WY'] = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
                                if wx_start is None: wx_start, wy_start = row['WX'], row['WY']

                                # --- 클럽 위치 예측 (칼만) ---
                                if kf_club is None:
                                    # 아직 한 번도 못 잡았으면 손목 위치를 기준으로 탐색 시작
                                    pred_x, pred_y = row['WX'], row['WY']
                                else:
                                    pred = kf_club.predict()
                                    pred_x, pred_y = float(pred[0, 0]), float(pred[1, 0])

                                # 놓친 프레임이 누적되면 탐색 반경을 점진적으로 넓혀 재포착 확률을 높임
                                cur_radius = roi_half * (1.0 + min(miss_streak, 6) * 0.25)

                                meas = detect_club_in_roi(frame, custom_model, pred_x, pred_y,
                                                           cur_radius, row['WX'], row['WY'],
                                                           conf_th, det_imgsz)

                                if meas is not None:
                                    mx, my = meas
                                    if kf_club is None:
                                        kf_club = create_kalman_2d(mx, my)
                                    else:
                                        kf_club.correct(np.array([[np.float32(mx)], [np.float32(my)]]))
                                    row['TX'], row['TY'] = mx, my
                                    row['T_Predicted'] = False
                                    miss_streak = 0
                                elif kf_club is not None:
                                    # 탐지 실패해도 직전 속도 기반 예측치로 채워서 궤적이 끊기지 않게 함
                                    row['TX'], row['TY'] = pred_x, pred_y
                                    row['T_Predicted'] = True
                                    miss_streak += 1
                                else:
                                    miss_streak += 1
                except Exception:
                    pass

                db_data.append(row)
            cap.release()

        with st.spinner("2단계: 데이터 정제 및 안정화 처리 중..."):
            df = pd.DataFrame(db_data)
            
            for col in ['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']:
                if col not in df.columns: df[col] = np.nan

            df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']] = df[['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']].interpolate(limit_direction='both')

            # 순간적으로 튀는 오탐(스파이크)을 중앙값 필터로 먼저 제거한 뒤 평균으로 부드럽게 처리
            for col in ['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']:
                df[col] = df[col].rolling(window=3, min_periods=1, center=True).median()

            for col in ['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY']:
                df[f'{col}_Smooth'] = df[col].rolling(window=5, min_periods=1, center=True).mean()

            for i in df.index:
                df.loc[i, 'SA_Smooth'] = get_blueprint_angle(df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], df.loc[i, 'TX_Smooth'], df.loc[i, 'TY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'LA_Smooth'] = get_blueprint_angle(df.loc[i, 'LX_Smooth'], df.loc[i, 'LY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'RA_Smooth'] = get_blueprint_angle(df.loc[i, 'RX_Smooth'], df.loc[i, 'RY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])

        with st.spinner("3단계: 시퀀스 타임라인 및 좌/우타 매칭 중..."):
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
        st.subheader("📸 청사진(Compass) 좌/우타 동기화 뷰")
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
                        draw_dynamic_visuals_with_compass(img, (int(row['RX_Smooth']), int(row['RY_Smooth'])), row['RA_Smooth'], ref_len*0.8, p1_gp[0], p1_gp[1], (0,255,0), "Rt Arm")

                status = "Pass"
                if row is not None and p['target'] is not None:
                    val = row['SA_Smooth'] if p['type'] == 'shaft' else (row['LA_Smooth'] if p['type'] == 'arm_left' else row['RA_Smooth'])
                    if not pd.isna(val) and angle_diff(val, p['target']) > 7.0: 
                        status = "Check"

                pred_tag = ""
                if row is not None and bool(row.get('T_Predicted', False)):
                    pred_tag = " · 클럽 예측값(칼만)"

                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{p['phase']}] {p['name']} ({status}){pred_tag}", use_column_width=True)
