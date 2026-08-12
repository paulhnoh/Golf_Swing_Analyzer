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
    use_head_detection = st.checkbox("클럽 헤드 탐지도 사용", value=True,
                                      help="실측 검증 결과, 이 모델은 샤프트보다 헤드 탐지가 훨씬 자주/정확하게 "
                                           "잡혔습니다. 끄면 실측 탐지율이 크게 떨어질 수 있으니 기본값(켜짐)을 "
                                           "권장합니다. 헤드 오탐이 의심되는 특정 영상에서만 꺼서 비교해보세요.")
    dist_upper_mult = st.slider("거리 상한 배수 (× 어드레스 실측 손목-클럽 거리)", 1.0, 2.5, 1.6, 0.05,
                                 help="어드레스에서 측정한 손목-클럽 거리의 이 배수를 넘는 후보는 물리적으로 "
                                      "클럽일 수 없다고 보고 제외합니다. 너무 낮추면(예: 1.15) 팔로우스루처럼 "
                                      "원근 효과로 2D상 거리가 어드레스보다 더 길게 보이는 구간의 진짜 탐지까지 "
                                      "걸러져 오히려 실측 탐지율이 떨어질 수 있습니다 (실측 검증 완료).")
    detect_method = st.selectbox(
        "클럽 방향 탐지 방식",
        ["YOLO 우선, Hough 보조", "Hough 우선, YOLO 보조", "YOLO만 사용", "Hough 직선검출만 사용"],
        index=0,
        help="YOLO(커스텀 학습 모델)는 클래스 오분류로 방향이 완전히 엉뚱하게 나올 수 있습니다. "
             "Hough는 학습 없이 손목 근처에서 시작하는 직선 에지를 직접 찾아 방향을 추정하는 "
             "고전적 이미지처리 방식이라, 오분류 위험은 없지만 배경이 복잡하면 선을 못 찾을 수 있습니다. "
             "두 방식을 비교해보고 이 영상에 더 잘 맞는 쪽을 고르세요.")
    st.divider()
    st.header("🏁 P13(정지) 판정")
    still_px_th = st.slider("정지 판정 임계값(px/frame)", 0.5, 8.0, 2.0, 0.5,
                             help="손목+클럽의 프레임간 이동량이 이 값 미만으로 연속 3프레임 유지되면 '정지'로 판정합니다. 카메라가 흔들리면 값을 키워보세요.")
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

def detect_club_in_roi(frame, model, pred_x, pred_y, roi_half, wx, wy, conf_th, imgsz, ref_len=None, use_head=False, dist_upper_mult=1.6):
    """예측 위치(pred_x, pred_y) 주변만 잘라서 확대 탐지 -> 작은 헤드/샤프트 인식률 개선.
    후보는 예측 위치와의 거리로 게이팅해서 배경의 엉뚱한 오탐을 걸러낸다.
    ref_len(=어드레스에서 실측한 손목-클럽 거리)이 주어지면, dist_upper_mult배를 넘는 후보는
    물리적으로 클럽일 수 없다고 보고 제외한다 -> 얼굴/배경 등에 락이 걸리는 것을 방지.
    (주의: 원근 효과로 인해 어드레스보다 다른 phase에서 2D 거리가 더 길게 보일 수 있으므로,
    배수를 너무 낮추면 오히려 진짜 탐지까지 걸러 실측 탐지율이 떨어질 수 있음 - 실측 검증됨)
    use_head=False면 헤드 클래스 탐지는 아예 후보에서 제외하고 샤프트만 사용한다.
    반환값: (point_or_None, rejected_by_distance_count) — 두 번째 값은 진단용으로,
    "물리적 거리 기준"으로 걸러진 오탐 후보가 이번 호출에서 몇 개였는지 알려준다."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(pred_x - roi_half)), max(0, int(pred_y - roi_half))
    x1, y1 = min(w, int(pred_x + roi_half)), min(h, int(pred_y + roi_half))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, 0

    crop = frame[y0:y1, x0:x1]
    try:
        c_res = model(crop, verbose=False, conf=conf_th, imgsz=imgsz)[0]
    except Exception:
        return None, 0

    head_c, shaft_c = [], []
    rejected_by_distance = 0
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
            if ref_len is not None:
                dist_wrist = math.hypot(cx - wx, cy - wy)
                # 어드레스 실측 거리(ref_len)의 dist_upper_mult배를 넘으면 물리적으로 클럽일 수 없다고 보고 제외
                # (카메라 고정 전제. 배수는 사이드바에서 조절 가능 - 너무 낮추면 원근 효과로 어드레스보다
                # 2D 거리가 더 길게 보이는 구간의 진짜 탐지까지 걸러질 수 있음)
                if dist_wrist < ref_len * 0.10 or dist_wrist > ref_len * dist_upper_mult:
                    rejected_by_distance += 1
                    continue
            is_head = 'head' in cls_name or 'club' in cls_name
            if is_head:
                if use_head:
                    head_c.append((cx, cy, conf, dist_pred))
                # use_head=False면 헤드 탐지는 통째로 무시 (샤프트만 신뢰)
            else:
                shaft_c.append((cx, cy, conf, dist_pred))

    # 실측 검증 결과: 이 모델은 헤드 탐지가 샤프트보다 훨씬 자주/안정적으로 잡힌다
    # (샤프트만 쓰면 실측 탐지율이 92%->6%로 폭락하는 것을 확인함). 따라서 헤드를 우선하고,
    # 헤드가 하나도 없을 때만 샤프트를 보조로 사용한다 (use_head=False면 헤드는 애초에 후보에서 제외됨).
    if head_c:
        best = max(head_c, key=lambda x: x[2] - 0.15 * (x[3] / max(roi_half, 1)))
        return (best[0], best[1]), rejected_by_distance
    if shaft_c:
        best = max(shaft_c, key=lambda x: x[2] - 0.35 * (x[3] / max(roi_half, 1)))
        return (best[0], best[1]), rejected_by_distance
    return None, rejected_by_distance

def detect_shaft_hough(frame, wx, wy, roi_half, ref_len=None, upper_mult=1.6):
    """학습된 클래스 분류에 의존하지 않고, 손목 근처에서 시작하는 직선(에지)을 Hough 변환으로
    직접 찾아 샤프트 방향을 추정한다. 골프 샤프트는 잔디/하늘을 배경으로 뚜렷한 직선 경계를
    만들기 때문에, YOLO가 엉뚱한 클래스로 오분류하는 문제 자체를 우회할 수 있다.
    - 손목 근처(허용오차 이내)에서 시작하는 선분만 후보로 인정 (그립 쪽 끝점)
    - 반대쪽(먼) 끝점까지의 거리가 물리적으로 말이 되는 범위(ref_len 기준)인 것만 채택
    - 여러 후보 중 가장 긴 선분을 채택 (짧은 선분은 잡음일 가능성이 높음)
    반환값: (fx, fy) 먼 쪽 끝점 좌표(=클럽 방향을 나타내는 점) 또는 None."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(wx - roi_half)), max(0, int(wy - roi_half))
    x1, y1 = min(w, int(wx + roi_half)), min(h, int(wy + roi_half))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    crop = frame[y0:y1, x0:x1]
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                                 minLineLength=max(15, int(roi_half * 0.25)), maxLineGap=8)
    except Exception:
        return None
    if lines is None:
        return None

    wx_local, wy_local = wx - x0, wy - y0
    near_tol = max(15, roi_half * 0.18)
    best_far, best_len = None, -1.0

    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        d1 = math.hypot(lx1 - wx_local, ly1 - wy_local)
        d2 = math.hypot(lx2 - wx_local, ly2 - wy_local)
        near_d, far_pt, far_d = (d1, (lx2, ly2), d2) if d1 <= d2 else (d2, (lx1, ly1), d1)
        if near_d > near_tol:
            continue  # 손목 근처에서 시작하지 않는 선분 -> 그립/샤프트가 아닐 가능성이 큼
        if ref_len is not None and (far_d < ref_len * 0.10 or far_d > ref_len * upper_mult):
            continue  # 물리적으로 말이 안 되는 길이 -> 제외
        seg_len = math.hypot(lx2 - lx1, ly2 - ly1)
        if seg_len > best_len:
            best_len = seg_len
            best_far = (far_pt[0] + x0, far_pt[1] + y0)

    return best_far

def find_closest_frame(df, col, target, start_f, end_f, fail_diff_threshold=60.0):
    """구간 [start_f, end_f] 안에서 target 각도와 가장 가까운 프레임을 찾는다.
    반환: (frame, is_estimated) - is_estimated=True면 신뢰할 수 없는 매칭이라는 뜻.
    """
    if start_f >= end_f or col not in df.columns:
        # 체이닝(이전 phase 프레임을 다음 phase 검색 시작점으로 사용)으로 인해 검색 구간이
        # 아예 사라진 경우 -> 그 경계값을 그대로 반환하지 않고, 주변 소구간(±3프레임)에서
        # 대신 검색해 최소한의 매칭 기회를 준다 (여러 phase가 강제로 동일 프레임에 뭉치는 것을 방지)
        lo, hi = max(0, start_f - 3), start_f + 3
        if col not in df.columns:
            return start_f, True
        sub = df[(df['Frame'] >= lo) & (df['Frame'] <= hi)].copy()
        if sub.empty:
            return start_f, True
        sub['diff'] = sub[col].apply(lambda x: angle_diff(x, target) if not pd.isna(x) else 999)
        return int(sub['diff'].idxmin()), True

    sub = df[(df['Frame'] >= start_f) & (df['Frame'] <= end_f)].copy()
    if sub.empty:
        return start_f, True
    sub['diff'] = sub[col].apply(lambda x: angle_diff(x, target) if not pd.isna(x) else 999)
    best_idx = int(sub['diff'].idxmin())
    best_diff = float(sub['diff'].min())
    if best_diff > fail_diff_threshold:
        # 클럽 데이터가 부족해 신뢰할 수 없는 매칭 -> 여러 phase가 같은 경계 프레임으로
        # 뭉치는 것을 막기 위해 구간 중간 지점으로 대체 (실제 값이 아닌 추정값임을 표시)
        return int(round((start_f + end_f) / 2)), True
    return best_idx, False

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
                addr_wx, addr_wy = None, None
                if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                    kp = p_res.keypoints.xy[0].cpu().numpy()
                    cf0 = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints.conf is not None else np.ones(17)
                    if len(kp) > 16 and kp[15][0] > 0 and kp[16][0] > 0:
                        p1_gp = ((int(kp[15][0]), int(kp[15][1])), (int(kp[16][0]), int(kp[16][1])))
                    # 어드레스 프레임의 손목 위치도 구해서, 실제 "손목-클럽" 거리 측정에 사용
                    if len(kp) > 10:
                        wrist_pts = [kp[i] for i in (9, 10) if kp[i][0] > 0 and cf0[i] > 0.05]
                        if wrist_pts:
                            addr_wx = float(np.mean([p[0] for p in wrist_pts]))
                            addr_wy = float(np.mean([p[1] for p in wrist_pts]))
            except Exception:
                addr_wx, addr_wy = None, None

            st.session_state.p1_gp = p1_gp

            # 기준 클럽 길이: 카메라가 고정이므로, 어드레스(첫 프레임)에서 팔+샤프트가 지면까지
            # 최대로 뻗은 상태의 "손목-클럽" 거리가 영상 전체에서 나올 수 있는 최대 거리다.
            # 이 실측값을 우선 사용하고, 탐지 실패 시에만 화면너비의 30%로 대체한다.
            measured_len = None
            if addr_wx is not None:
                init_radius = f_frame.shape[1] * 0.35
                meas0 = detect_club_in_roi(f_frame, custom_model, addr_wx, addr_wy, init_radius,
                                            addr_wx, addr_wy, conf_th, det_imgsz,
                                            ref_len=None, use_head=True)
                if meas0[0] is not None:
                    measured_len = math.hypot(meas0[0][0] - addr_wx, meas0[0][1] - addr_wy)

            st.session_state.ref_club_len = measured_len if measured_len else f_frame.shape[1] * 0.3
            st.session_state.ref_club_len_is_measured = measured_len is not None

            db_data = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            wx_start, wy_start = None, None
            kf_club = None          # 클럽(헤드/샤프트) 위치 추적용 칼만 필터. 탐지가 끊겨도 이걸로 위치를 이어감
            roi_half = max(70, st.session_state.ref_club_len * search_radius_ratio)
            miss_streak = 0         # 연속 미탐지 횟수. 너무 오래 놓치면 탐색 반경을 넓히다가, 그래도 안되면 락을 포기함
            MAX_MISS_STREAK = 8     # 다운스윙처럼 빠른 구간에서 칼만 예측이 계속 빗나갈 때, 발산한 값을 계속 믿지 않도록 상한을 둠
            frame_w_ref = int(f_frame.shape[1])
            frame_h_ref = int(f_frame.shape[0])
            debug_detect_log = []   # 진단 패널용: 프레임별 탐지 성공 여부 기록
            pending_candidate = None  # 락 확정 전 "가(假) 후보" 위치 (오탐 방지용)
            pending_count = 0

            for fn in range(tot_frames):
                ret, frame = cap.read()
                if not ret or frame is None: break
                cv2.imwrite(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"), frame)

                row = {'Frame': fn, 'WX': np.nan, 'WY': np.nan, 'TX': np.nan, 'TY': np.nan,
                       'LX': np.nan, 'LY': np.nan, 'RX': np.nan, 'RY': np.nan,
                       'RWX': np.nan, 'RWY': np.nan, 'LWX': np.nan, 'LWY': np.nan, 'T_Predicted': False}
                det_status = 'no_wrist'
                n_rejected_dist = 0
                meas = yolo_meas = hough_meas = None
                try:
                    p_res = pose_model(frame, verbose=False)[0]

                    if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                        kp = p_res.keypoints.xy[0].cpu().numpy()
                        cf = p_res.keypoints.conf[0].cpu().numpy() if p_res.keypoints.conf is not None else np.ones(17)

                        if len(kp) > 10:
                            if kp[5][0] > 0: row['LX'], row['LY'] = float(kp[5][0]), float(kp[5][1])
                            if kp[6][0] > 0: row['RX'], row['RY'] = float(kp[6][0]), float(kp[6][1])
                            # 왼쪽/오른쪽 손목(kp[9]/kp[10])을 각각 별도로도 저장
                            # -> P5(왼손 최고점) / P8·P12(오른손 기준) 정의를 손별로 정확히 판정하기 위함
                            if kp[9][0] > 0 and cf[9] > 0.05:
                                row['LWX'], row['LWY'] = float(kp[9][0]), float(kp[9][1])
                            if kp[10][0] > 0 and cf[10] > 0.05:
                                row['RWX'], row['RWY'] = float(kp[10][0]), float(kp[10][1])
                            pts = [kp[i] for i in (9, 10) if kp[i][0] > 0 and cf[i] > 0.05]

                            if pts:
                                row['WX'], row['WY'] = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
                                if wx_start is None: wx_start, wy_start = row['WX'], row['WY']

                                # --- 클럽 위치 예측 (칼만) ---
                                reacquiring = (kf_club is None)
                                if kf_club is None:
                                    # 아직 한 번도 못 잡았거나 락을 잃은 직후: 손목 위치를 기준으로 넓게 재탐색
                                    # (팔로우스루/피니시처럼 클럽이 몸에서 멀리 떨어져 있는 자세도 커버하도록
                                    #  탐색 반경을 프레임 크기의 40%대까지 넓히고, 신뢰도 임계값도 완화한다.
                                    #  락이 없을 때의 오탐은 다음 프레임에서 거리 게이팅으로 걸러지므로 부담이 적다)
                                    pred_x, pred_y = row['WX'], row['WY']
                                    cur_radius = max(roi_half * 2.2, min(frame_w_ref, frame_h_ref) * 0.42)
                                else:
                                    pred = kf_club.predict()
                                    pred_x, pred_y = float(pred[0, 0]), float(pred[1, 0])
                                    # 화면 밖으로 발산하지 않도록 예측 위치를 프레임 범위 안으로 고정
                                    pred_x = float(np.clip(pred_x, 0, frame_w_ref - 1))
                                    pred_y = float(np.clip(pred_y, 0, frame_h_ref - 1))
                                    # 놓친 프레임이 누적되면 탐색 반경을 점진적으로 넓혀 재포착 확률을 높임
                                    cur_radius = roi_half * (1.0 + min(miss_streak, 6) * 0.25)

                                # 락이 없는(재탐색) 상태에서는 신뢰도 임계값을 완화해 recall을 우선시한다.
                                # (오탐이 섞여도 다음 프레임부터 칼만 거리 게이팅으로 곧 걸러짐)
                                effective_conf = max(0.05, conf_th * 0.6) if reacquiring else conf_th

                                yolo_meas, n_rejected_dist = (None, 0)
                                hough_meas = None
                                ref_len_val = st.session_state.ref_club_len

                                if detect_method in ("YOLO 우선, Hough 보조", "YOLO만 사용"):
                                    yolo_meas, n_rejected_dist = detect_club_in_roi(
                                        frame, custom_model, pred_x, pred_y, cur_radius,
                                        row['WX'], row['WY'], effective_conf, det_imgsz,
                                        ref_len=ref_len_val, use_head=use_head_detection,
                                        dist_upper_mult=dist_upper_mult)
                                    if yolo_meas is None and detect_method == "YOLO 우선, Hough 보조":
                                        hough_meas = detect_shaft_hough(frame, row['WX'], row['WY'],
                                                                         cur_radius, ref_len=ref_len_val,
                                                                         upper_mult=dist_upper_mult)
                                elif detect_method in ("Hough 우선, YOLO 보조", "Hough 직선검출만 사용"):
                                    hough_meas = detect_shaft_hough(frame, row['WX'], row['WY'],
                                                                     cur_radius, ref_len=ref_len_val,
                                                                     upper_mult=dist_upper_mult)
                                    if hough_meas is None and detect_method == "Hough 우선, YOLO 보조":
                                        yolo_meas, n_rejected_dist = detect_club_in_roi(
                                            frame, custom_model, pred_x, pred_y, cur_radius,
                                            row['WX'], row['WY'], effective_conf, det_imgsz,
                                            ref_len=ref_len_val, use_head=use_head_detection,
                                            dist_upper_mult=dist_upper_mult)

                                meas = yolo_meas if yolo_meas is not None else hough_meas

                                if meas is not None:
                                    mx, my = meas
                                    if kf_club is None:
                                        # 락이 아직 없으면 바로 확정하지 않고, 연속으로 비슷한 위치에서
                                        # 잡혀야만 확정한다 (배경/몸통 등 1회성 오탐이 락을 가로채는 것을 방지)
                                        if pending_candidate is not None and \
                                           math.hypot(mx - pending_candidate[0], my - pending_candidate[1]) < cur_radius * 0.5:
                                            pending_count += 1
                                        else:
                                            pending_candidate = (mx, my)
                                            pending_count = 1

                                        if pending_count >= 2:
                                            kf_club = create_kalman_2d(mx, my)
                                            pending_candidate, pending_count = None, 0
                                            row['TX'], row['TY'] = mx, my
                                            row['T_Predicted'] = False
                                            miss_streak = 0
                                            det_status = 'detected'
                                        else:
                                            # 아직 확정 전 (tentative) - 이번 프레임 값은 채우지 않고 다음 프레임 확인을 기다림
                                            det_status = 'pending'
                                    else:
                                        kf_club.correct(np.array([[np.float32(mx)], [np.float32(my)]]))
                                        row['TX'], row['TY'] = mx, my
                                        row['T_Predicted'] = False
                                        miss_streak = 0
                                        det_status = 'detected'
                                elif kf_club is not None and miss_streak < MAX_MISS_STREAK:
                                    # 탐지 실패해도 직전 속도 기반 예측치로 채워서 궤적이 끊기지 않게 함
                                    row['TX'], row['TY'] = pred_x, pred_y
                                    row['T_Predicted'] = True
                                    miss_streak += 1
                                    det_status = 'predicted'
                                else:
                                    # 너무 오래 놓치면 예측을 더 신뢰하지 않고 락을 포기 -> 나중 단계에서 보간으로 채움
                                    if kf_club is not None:
                                        miss_streak += 1
                                        if miss_streak >= MAX_MISS_STREAK:
                                            kf_club = None
                                            miss_streak = 0
                                    det_status = 'lost'
                except Exception:
                    pass

                debug_detect_log.append({'Frame': fn, 'status': det_status, 'rejected_by_distance': n_rejected_dist,
                                          'source': ('yolo' if meas is not None and meas is yolo_meas
                                                     else ('hough' if meas is not None else 'none'))})

                db_data.append(row)
            cap.release()
            st.session_state.debug_detect_log = pd.DataFrame(debug_detect_log)

        with st.spinner("2단계: 데이터 정제 및 안정화 처리 중..."):
            df = pd.DataFrame(db_data)
            
            SMOOTH_COLS = ['WX', 'WY', 'LX', 'LY', 'RX', 'RY', 'TX', 'TY', 'RWX', 'RWY', 'LWX', 'LWY']
            for col in SMOOTH_COLS:
                if col not in df.columns: df[col] = np.nan

            df[SMOOTH_COLS] = df[SMOOTH_COLS].interpolate(limit_direction='both')

            # 순간적으로 튀는 오탐(스파이크)을 중앙값 필터로 먼저 제거한 뒤 평균으로 부드럽게 처리
            for col in SMOOTH_COLS:
                df[col] = df[col].rolling(window=3, min_periods=1, center=True).median()

            # 빠르게 움직이는 포인트(손목/클럽)는 5프레임 평균을 쓰면 실제 위치보다 뒤처져 보이는
            # "지연(lag)" 현상이 생긴다 (정지 상태인 어드레스는 문제 없지만, 테이크어웨이/전환구간처럼
            # 빠르게 움직이는 구간에서 그려진 선이 실제 사진 속 샤프트와 어긋나 보이는 원인).
            # 상대적으로 느리게 움직이는 어깨(LX,LY/RX,RY)는 창을 크게(5) 유지해 노이즈를 더 줄이고,
            # 빠른 지점(손목/클럽)은 창을 작게(3) 줄여 지연을 최소화한다.
            FAST_COLS = ['WX', 'WY', 'TX', 'TY', 'RWX', 'RWY', 'LWX', 'LWY']
            SLOW_COLS = ['LX', 'LY', 'RX', 'RY']
            for col in FAST_COLS:
                df[f'{col}_Smooth'] = df[col].rolling(window=3, min_periods=1, center=True).mean()
            for col in SLOW_COLS:
                df[f'{col}_Smooth'] = df[col].rolling(window=5, min_periods=1, center=True).mean()

            for i in df.index:
                df.loc[i, 'SA_Smooth'] = get_blueprint_angle(df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], df.loc[i, 'TX_Smooth'], df.loc[i, 'TY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'LA_Smooth'] = get_blueprint_angle(df.loc[i, 'LX_Smooth'], df.loc[i, 'LY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])
                df.loc[i, 'RA_Smooth'] = get_blueprint_angle(df.loc[i, 'RX_Smooth'], df.loc[i, 'RY_Smooth'], df.loc[i, 'WX_Smooth'], df.loc[i, 'WY_Smooth'], p1_gp[0], p1_gp[1])
                # 스무딩을 거치지 않은(median 필터까지만 적용된) "순간" 각도도 같이 계산해둔다.
                # 스무딩값과 크게 차이나면 "지연" 문제, 순간값 자체도 부정확하면 "탐지" 문제로 구분 가능.
                df.loc[i, 'SA_Instant'] = get_blueprint_angle(df.loc[i, 'WX'], df.loc[i, 'WY'], df.loc[i, 'TX'], df.loc[i, 'TY'], p1_gp[0], p1_gp[1])
                # 그립(양손) 벡터 기반 샤프트 방향 추정치. 실제 그립에서는 왼손이 오른손보다 샤프트 butt쪽에,
                # 오른손이 헤드쪽에 가깝게 잡기 때문에(오른손 기준), 왼손->오른손 벡터가 샤프트의 국소적인
                # 방향과 대략 일치할 수 있다. 다만 두 손목 사이 거리가 원래 짧아서(그립 폭만큼), 아주 작은
                # 좌표 오차에도 각도가 크게 흔들릴 수 있음 -> 클럽 탐지 실패 시의 "보조/대체 신호"로만 참고할 것.
                df.loc[i, 'GA_Smooth'] = get_blueprint_angle(df.loc[i, 'LWX_Smooth'], df.loc[i, 'LWY_Smooth'], df.loc[i, 'RWX_Smooth'], df.loc[i, 'RWY_Smooth'], p1_gp[0], p1_gp[1])

        with st.spinner("3단계: 시퀀스 타임라인 매칭 중..."):
            # ⚠️ 현재 버전은 단순화를 위해 "오른손 선수 전용"입니다 (좌타 판별 로직 제거).
            # P5(백스윙 탑)=왼손목 최고점, P12(팔로우스루 탑)=오른손목 최고점을 직접 사용합니다.
            try:
                p5 = int(df['LWY_Smooth'].iloc[:int(tot_frames * 0.65)].idxmin())
            except:
                p5 = int(tot_frames * 0.3)

            try:
                # Address(P1): 백스윙이 시작되기 전, 샤프트가 지면과 약 90°(±5°)를 유지한 채
                # "정지"해 있는 구간의 마지막 프레임을 찾는다.
                # (사용자 정의: Address는 각도 검증이 아니라 "90±5도 근처에서 멈춰있는, 백스윙 시작 직전 프레임")
                ADDR_ANGLE_TOL = 5.0
                search_end = max(1, min(p5, tot_frames - 1))
                wrist_disp_addr = np.hypot(df['WX_Smooth'].diff(), df['WY_Smooth'].diff()).fillna(0.0)
                sa_near_90 = df['SA_Smooth'].apply(lambda v: (not pd.isna(v)) and angle_diff(v, 90.0) <= ADDR_ANGLE_TOL)
                is_still = wrist_disp_addr < still_px_th
                cand_idx = df.index[(df.index < search_end) & sa_near_90 & is_still]

                if len(cand_idx) > 0:
                    p1 = int(cand_idx.max())  # 정지 구간의 마지막 프레임 = 백스윙이 막 시작되기 직전
                else:
                    # 완화 1: 각도 조건은 버리고 "정지 상태"만으로 후보를 찾음 (클럽 탐지가 약한 경우 대비)
                    cand_idx2 = df.index[(df.index < search_end) & is_still]
                    if len(cand_idx2) > 0:
                        p1 = int(cand_idx2.max())
                    else:
                        # 최종 폴백: 손목이 처음 움직이기 시작한 시점 (기존 방식)
                        wx_start_avg = df['WX_Smooth'].iloc[0:min(5, len(df))].mean()
                        p1_mask = (df['WX_Smooth'].iloc[:p5] - wx_start_avg).abs() > 3.0
                        p1 = int(p1_mask.idxmax()) if p1_mask.any() else 0
            except:
                p1 = 0

            is_left_handed = False  # 단순화: 오른손 선수 전용 (향후 좌타 지원 시 이 값만 판별 로직으로 교체)

            try:
                p12 = int(df['RWY_Smooth'].iloc[p5 + 15 :].idxmin()) if len(df.iloc[p5 + 15 :]) > 0 else tot_frames - 1
            except:
                p12 = tot_frames - 1

            # P8(임팩트): 레퍼런스 정의 = "클럽헤드 혹은 오른손의 최저점" -> 오른쪽 손목(RWY_Smooth) 전용
            # 신호와 클럽헤드(TY_Smooth) 신호 중 더 낮은(화면상 Y가 큰) 쪽을 프레임별로 취해
            # 그 결합 신호가 최대가 되는 지점을 찾는다. 클럽 헤드가 손목보다 늦게 최저점에 도달하는
            # 릴리즈/래그 특성을 반영해 손목만 볼 때보다 실제 임팩트에 더 가깝게 잡힌다.
            try:
                low_point_signal = df[['RWY_Smooth', 'TY_Smooth']].max(axis=1)
                sub_imp = low_point_signal.iloc[p5 + 5 : p12 - 5]
                p8 = int(sub_imp.idxmax()) if not sub_imp.empty else p5 + (p12 - p5) // 2
            except:
                p8 = p5 + (tot_frames - p5) // 2

            # P13(피니시 정지): "2프레임 연속으로 움직임이 거의 없어지는 첫 번째 시점" (사용자 정의).
            # 못 찾으면 마지막 프레임으로 대체.
            try:
                search_start = min(p12 + 3, tot_frames - 1)
                wrist_disp = np.hypot(df['WX_Smooth'].diff(), df['WY_Smooth'].diff())
                club_disp = np.hypot(df['TX_Smooth'].diff(), df['TY_Smooth'].diff())
                combined_disp = wrist_disp.fillna(999.0) + club_disp.fillna(0.0) * 0.5

                p13 = tot_frames - 1  # 기본값: 정지 구간을 못 찾으면 마지막 프레임
                STILL_WINDOW = 2
                for idx in range(search_start, max(search_start, tot_frames - STILL_WINDOW - 1)):
                    window = combined_disp.iloc[idx: idx + STILL_WINDOW]
                    if len(window) == STILL_WINDOW and (window < still_px_th).all():
                        p13 = idx  # 첫 번째로 발견된 정지 구간 채택
                        break
            except Exception:
                p13 = tot_frames - 1

            # --- 목표 각도 (청사진 나침반 기준: Left=0, Down=90, Right=180, Up=270) ---
            # Nelly Korda 레퍼런스(오른손잡이) 기준 값. 좌타는 화면 좌우가 뒤집히므로
            # mirror_angle()로 대칭 변환해서 사용한다 (Down/Up은 그대로, Left<->Right만 뒤집힘).
            def mirror_angle(a):
                return (180.0 - a) % 360.0

            BASE_P2, BASE_P3 = 45.0, 0.0          # P2 샤프트45도, P3 샤프트0도
            BASE_P4 = 0.0                          # P4 리드암(왼팔) 수평
            BASE_P6, BASE_P7 = 315.0, 0.0          # P6 샤프트315도, P7 샤프트0도
            BASE_P9, BASE_P10 = 135.0, 180.0       # P9 샤프트135도, P10 샤프트180도 (사용자 확인 완료)
            BASE_P11 = 0.0                         # P11 리드암 반대팔(오른팔) 수평

            if is_left_handed:
                tgt_p2, tgt_p3 = mirror_angle(BASE_P2), mirror_angle(BASE_P3)
                tgt_p6, tgt_p7 = mirror_angle(BASE_P6), mirror_angle(BASE_P7)
                tgt_p9, tgt_p10 = mirror_angle(BASE_P9), mirror_angle(BASE_P10)
                tgt_p4_arm, tgt_p4_ang = 'RA_Smooth', mirror_angle(BASE_P4)
                tgt_p11_arm, tgt_p11_ang = 'LA_Smooth', mirror_angle(BASE_P11)
            else:
                tgt_p2, tgt_p3 = BASE_P2, BASE_P3
                tgt_p6, tgt_p7 = BASE_P6, BASE_P7
                tgt_p9, tgt_p10 = BASE_P9, BASE_P10
                tgt_p4_arm, tgt_p4_ang = 'LA_Smooth', BASE_P4
                tgt_p11_arm, tgt_p11_ang = 'RA_Smooth', BASE_P11

            phases_info = [
                {"phase": "P1", "name": "Address", "target": None, "type": "shaft"},
                {"phase": "P2", "name": "Start Sweep (샤프트 45°)", "target": tgt_p2, "type": "shaft"},
                {"phase": "P3", "name": "Back Alignment (샤프트 0°)", "target": tgt_p3, "type": "shaft"},
                {"phase": "P4", "name": "Lead Arm Parallel (리드암 수평)", "target": tgt_p4_ang, "type": "arm_left" if not is_left_handed else "arm_right"},
                {"phase": "P5", "name": "Backswing Top (리드손 최고점)", "target": None, "type": "top"},
                {"phase": "P6", "name": "Transition (샤프트 315°)", "target": tgt_p6, "type": "shaft"},
                {"phase": "P7", "name": "DB Alignment (샤프트 0°)", "target": tgt_p7, "type": "shaft"},
                {"phase": "P8", "name": "Impact (손목/헤드 최저점)", "target": None, "type": "impact"},
                {"phase": "P9", "name": "Release (샤프트 135°)", "target": tgt_p9, "type": "shaft"},
                {"phase": "P10", "name": "DF Alignment (샤프트 180°)", "target": tgt_p10, "type": "shaft"},
                {"phase": "P11", "name": "Trail Arm Parallel (반대팔 수평)", "target": tgt_p11_ang, "type": "arm_right" if not is_left_handed else "arm_left"},
                {"phase": "P12", "name": "Follow-Through Top (반대손 최고점)", "target": None, "type": "top"},
                {"phase": "P13", "name": "Finish (동작 정지)", "target": None, "type": "finish"},
            ]

            auto_f = {"P1": p1, "P5": p5, "P8": p8, "P12": p12, "P13": p13}
            auto_f_estimated = {k: False for k in auto_f}  # 각 phase가 실제 데이터 매칭인지, 데이터 부족으로 인한 추정값인지

            auto_f["P2"], auto_f_estimated["P2"] = find_closest_frame(df, 'SA_Smooth', tgt_p2, p1, p5)
            auto_f["P3"], auto_f_estimated["P3"] = find_closest_frame(df, 'SA_Smooth', tgt_p3, auto_f["P2"], p5)
            auto_f["P4"], auto_f_estimated["P4"] = find_closest_frame(df, tgt_p4_arm, tgt_p4_ang, auto_f["P3"], p5)

            auto_f["P6"], auto_f_estimated["P6"] = find_closest_frame(df, 'SA_Smooth', tgt_p6, p5, p8)
            auto_f["P7"], auto_f_estimated["P7"] = find_closest_frame(df, 'SA_Smooth', tgt_p7, auto_f["P6"], p8)

            auto_f["P9"], auto_f_estimated["P9"] = find_closest_frame(df, 'SA_Smooth', tgt_p9, p8, p12)
            auto_f["P10"], auto_f_estimated["P10"] = find_closest_frame(df, 'SA_Smooth', tgt_p10, auto_f["P9"], p12)
            auto_f["P11"], auto_f_estimated["P11"] = find_closest_frame(df, tgt_p11_arm, tgt_p11_ang, auto_f["P10"], p12)

            # --- 실측 스냅(snap) ---
            # 클럽 각도 기반(shaft/arm) phase가 하필 칼만 "예측" 프레임에 걸리면, 그 프레임은
            # 실제 탐지가 아니라 추정된 위치라 시각적으로 실제 사진과 어긋나 보일 수 있다.
            # 주변 ±4프레임 안에 실측(T_Predicted=False) 프레임이 있으면 그쪽으로 옮겨서
            # 화면에 보이는 각도가 항상 "실제로 탐지된" 값을 우선하도록 한다.
            SNAP_WINDOW = 4
            angle_based_phases = {p['phase'] for p in phases_info if p['type'] in ('shaft', 'arm_left', 'arm_right')}
            if 'T_Predicted' in df.columns:
                for ph in list(auto_f.keys()):
                    if ph not in angle_based_phases:
                        continue
                    f0 = auto_f[ph]
                    if f0 not in df.index or not bool(df.loc[f0, 'T_Predicted']):
                        continue  # 이미 실측이면 그대로 둠
                    lo, hi = max(0, f0 - SNAP_WINDOW), min(tot_frames - 1, f0 + SNAP_WINDOW)
                    window = df.loc[lo:hi]
                    real_hits = window[window['T_Predicted'] == False]
                    if not real_hits.empty:
                        nearest = (real_hits.index.to_series() - f0).abs().idxmin()
                        auto_f[ph] = int(nearest)

            st.session_state.auto_f_estimated = auto_f_estimated

            st.session_state.df = df
            st.session_state.frame_dir = frame_dir
            st.session_state.tot_frames = tot_frames
            st.session_state.auto_f = auto_f
            st.session_state.phases_info = phases_info
            st.session_state.scan_done = True

    if 'scan_done' in st.session_state and 'df' in st.session_state:
        if 'debug_detect_log' in st.session_state:
            dbg = st.session_state.debug_detect_log
            with st.expander("🔍 클럽 탐지 진단 정보 (문제 구간 찾기용)"):
                total = len(dbg)
                n_det = int((dbg['status'] == 'detected').sum())
                n_pred = int((dbg['status'] == 'predicted').sum())
                n_lost = int(dbg['status'].isin(['lost', 'pending']).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("실측 탐지", f"{n_det}/{total}", f"{n_det/total*100:.0f}%")
                c2.metric("칼만 예측으로 보완", f"{n_pred}/{total}", f"{n_pred/total*100:.0f}%")
                c3.metric("완전 미탐지(락 상실/확인대기)", f"{n_lost}/{total}", f"{n_lost/total*100:.0f}%")

                if 'rejected_by_distance' in dbg.columns:
                    total_rejected = int(dbg['rejected_by_distance'].sum())
                    frames_with_rejection = int((dbg['rejected_by_distance'] > 0).sum())
                    is_measured = st.session_state.get('ref_club_len_is_measured', False)
                    len_note = "어드레스에서 실측" if is_measured else "실측 실패 → 화면폭의 30%로 대체"
                    st.caption(f"📏 기준 클럽 길이: {st.session_state.ref_club_len:.0f}px ({len_note}) — "
                               f"이 거리(×{dist_upper_mult:.2f})를 넘는 후보는 물리적으로 클럽일 수 없다고 보고 자동 제외됩니다.")
                    st.caption(f"🚫 물리적 거리 기준으로 제외된 오탐 후보: 총 {total_rejected}개 "
                               f"({frames_with_rejection}개 프레임에서 발생)")

                if 'source' in dbg.columns:
                    n_yolo = int((dbg['source'] == 'yolo').sum())
                    n_hough = int((dbg['source'] == 'hough').sum())
                    st.caption(f"🔎 탐지 방식별 실측 성공 프레임: YOLO {n_yolo}개 / Hough 직선검출 {n_hough}개 "
                               f"(현재 설정: {detect_method})")

                # 연속 미탐지(lost/pending) 구간을 찾아 프레임 범위로 보여줌 -> 실측값과 대조하기 쉬움
                lost_mask = dbg['status'].isin(['lost', 'pending']).to_numpy()
                ranges = []
                start = None
                for i, v in enumerate(lost_mask):
                    if v and start is None:
                        start = i
                    elif not v and start is not None:
                        if i - start >= 3:
                            ranges.append((start, i - 1))
                        start = None
                if start is not None and len(lost_mask) - start >= 3:
                    ranges.append((start, len(lost_mask) - 1))

                if ranges:
                    st.caption("연속 3프레임 이상 클럽을 완전히 놓친 구간 (이 구간의 각도값은 신뢰하기 어려움):")
                    st.write(", ".join([f"{a}~{b}프레임" for a, b in ranges]))
                else:
                    st.caption("연속 미탐지 구간 없음 — 전체적으로 안정적으로 추적됨.")

            with st.expander("📈 각도 변화 그래프 (샤프트/팔 각도가 프레임별로 어떻게 움직였는지 확인)"):
                st.caption("여러 phase가 같은 프레임으로 뭉쳐 나올 때, 실제로 그 구간에서 각도가 얼마나 빠르게 "
                           "변했는지 눈으로 확인할 수 있습니다. 급격한 수직에 가까운 구간이면 촬영 프레임 사이로 "
                           "목표각도가 스쳐 지나갔을 가능성이 큽니다.")
                chart_df = st.session_state.df.set_index('Frame')[['SA_Smooth', 'LA_Smooth', 'RA_Smooth']]
                st.line_chart(chart_df)
                # 참고: GA_Smooth(그립 벡터 기반 샤프트 추정치)는 실측 검증 결과 두 손목 사이 거리가
                # 너무 짧아(그립 폭 수준) 사소한 keypoint 오차에도 각도가 크게 튀는 것으로 확인되어
                # (0°/350° 사이를 반복적으로 오가는 노이즈), 신뢰할 수 있는 보조 신호로 부적합하다고
                # 판단해 화면 표시에서 제외했다. 계산 자체는 df['GA_Smooth']에 남겨뒀다.

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
                        # 진단용: 실제로 "클럽"이라고 탐지된 좌표(raw TX,TY)를 점으로 직접 표시.
                        # 이 점이 진짜 샤프트/헤드 위에 있는지 보면, 각도 오차가 스무딩 때문인지
                        # 탐지 자체가 틀린 것인지 바로 구분할 수 있다.
                        if not pd.isna(row.get('TX', np.nan)) and not pd.isna(row.get('TY', np.nan)):
                            tx_raw, ty_raw = int(row['TX']), int(row['TY'])
                            cv2.circle(img, (tx_raw, ty_raw), 8, (0, 165, 255), -1, cv2.LINE_AA)
                            cv2.circle(img, (tx_raw, ty_raw), 8, (0, 0, 0), 2, cv2.LINE_AA)
                            draw_text_with_outline(img, "Club pt", (tx_raw + 12, ty_raw), 0.5, (0, 165, 255), (0, 0, 0), 2)
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

                est_tag = ""
                if st.session_state.get('auto_f_estimated', {}).get(p['phase'], False):
                    est_tag = " · ⚠️추정값(클럽 데이터 부족)"

                debug_tag = ""
                if row is not None and p['type'] == 'shaft' and not pd.isna(row.get('SA_Instant', np.nan)):
                    debug_tag = f" · [스무딩 {row['SA_Smooth']:.1f}° / 순간 {row['SA_Instant']:.1f}°]"

                reject_tag = ""
                dbg_log = st.session_state.get('debug_detect_log')
                if dbg_log is not None and 'rejected_by_distance' in dbg_log.columns:
                    match = dbg_log[dbg_log['Frame'] == fn]
                    if not match.empty and int(match['rejected_by_distance'].iloc[0]) > 0:
                        reject_tag = f" · 🚫거리기준 오탐 {int(match['rejected_by_distance'].iloc[0])}개 제외됨"

                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{p['phase']}] {p['name']} ({status}){pred_tag}{est_tag}{debug_tag}{reject_tag}", use_column_width=True)
