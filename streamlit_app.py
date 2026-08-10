import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import os
import tempfile
from ultralytics import YOLO

# 1. 시스템 설정
st.set_page_config(page_title="P1-P13 Ultimate Full Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 전체 통합 정밀 분석 시스템")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

# 분석 페이즈 정의
phases_info = [
    {"phase": "P1", "name": "Address", "desc": "샤프트 지면 수직", "target_angle": 90.0, "type": "shaft"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트 45°", "target_angle": 45.0, "type": "shaft"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트 우측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔 우측 수평", "target_angle": 0.0, "type": "arm_left"},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (탑)", "target_angle": None, "type": "top"},
    {"phase": "P6", "name": "Transition", "desc": "샤프트 다운스윙 315°", "target_angle": 315.0, "type": "shaft"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트 우측 수평", "target_angle": 0.0, "type": "shaft"},
    {"phase": "P8", "name": "Impact", "desc": "볼 타격 (최저점)", "target_angle": None, "type": "impact"},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트 릴리스 135°", "target_angle": 135.0, "type": "shaft"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트 좌측 수평", "target_angle": 180.0, "type": "shaft"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔 좌측 수평", "target_angle": 180.0, "type": "arm_right"},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (피니시 진입)", "target_angle": None, "type": "top"},
    {"phase": "P13", "name": "Finish", "desc": "스윙 종료 정지", "target_angle": None, "type": "finish"},
]

def compute_relative_angle(p1, p2, ground_p1, ground_p2):
    if ground_p1[0] > ground_p2[0]: ground_p1, ground_p2 = ground_p2, ground_p1
    g_dx, g_dy = ground_p2[0]-ground_p1[0], ground_p2[1]-ground_p1[1]
    ground_tilt = math.atan2(g_dy, g_dx)
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    cos_t, sin_t = math.cos(-ground_tilt), math.sin(-ground_tilt)
    rx = -(dx * cos_t - dy * sin_t)
    ry = dx * sin_t + dy * cos_t
    angle = math.degrees(math.atan2(ry, rx))
    return round(angle % 360, 1)

def draw_text_with_outline(img, text, pos, font_scale, text_color, outline_color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_angle_visual(img, vertex, target_pt, measured_val, ground_p1, ground_p2, color, label):
    if pd.isna(measured_val): return
    g_dx, g_dy = ground_p1[0]-ground_p2[0], ground_p1[1]-ground_p2[1]
    ground_tilt_deg = math.degrees(math.atan2(-g_dy, -g_dx))
    
    cv2.line(img, vertex, target_pt, color, 4, cv2.LINE_AA)
    mid_a = measured_val / 2.0
    mid_rad = math.radians(mid_a + ground_tilt_deg)
    txt_x, txt_y = int(vertex[0] + 70 * math.cos(mid_rad)), int(vertex[1] + 70 * math.sin(mid_rad))
    draw_text_with_outline(img, f"{label}: {measured_val}°", (txt_x-30, txt_y+5), 0.7, (0, 255, 255), (0, 0, 0), 2)

uploaded_file = st.file_uploader("스윙 영상 업로드", type=['mp4', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    frame_dir = tempfile.mkdtemp()
    
    # 2. 데이터베이스 스캔
    db_records = []
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    with st.spinner("분석 중..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_path = os.path.join(frame_dir, f"frame_{fn:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            p_res = pose_model(frame, verbose=False)[0]
            c_res = custom_model(frame, verbose=False)[0]
            
            # 좌표 추출 (손목, 샤프트 끝, 어깨)
            row = {'Frame': fn}
            if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                kpts = p_res.keypoints.xy[0].cpu().numpy()
                row['WristX'] = np.mean([kpts[9][0], kpts[10][0]])
                row['WristY'] = np.mean([kpts[9][1], kpts[10][1]])
                row['LShoulderX'], row['LShoulderY'] = kpts[5]
                row['RShoulderX'], row['RShoulderY'] = kpts[6]
                
                # 샤프트 추적
                best_target = None
                for box in c_res.boxes:
                    name = c_res.names[int(box.cls[0])]
                    if name in ['head', 'shaft']:
                        cent = (int((box.xyxy[0][0]+box.xyxy[0][2])/2), int((box.xyxy[0][1]+box.xyxy[0][3])/2))
                        if best_target is None or math.hypot(cent[0]-row['WristX'], cent[1]-row['WristY']) > math.hypot(best_target[0]-row['WristX'], best_target[1]-row['WristY']):
                            best_target = cent
                if best_target:
                    row['TargetX'], row['TargetY'] = best_target
                    
            db_records.append(row)
        cap.release()

    df = pd.DataFrame(db_records).interpolate(method='linear', limit_direction='both')
    
    # 3. W-Curve 기준 자동 뼈대 매핑
    p8 = int(df['WristY'].idxmax()) # 임팩트
    p5 = int(df.loc[:p8]['WristY'].idxmin()) # 백스윙 탑
    p1 = 0
    p12 = int(df.loc[p8:]['WristY'].idxmin()) # 피니시 진입
    
    # 4. 시각화 UI
    st.subheader("📸 초정밀 스윙 분석 결과")
    cols = st.columns(4)
    for i, phase in enumerate(phases_info):
        with cols[i % 4]:
            # 각 페이즈별 프레임 결정
            fn = st.slider(f"[{phase['phase']}] 조정", 0, len(df)-1, key=f"s_{i}")
            
            img = cv2.imread(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"))
            row = df.loc[fn]
            
            # 오버레이 그리기
            if phase['type'] == 'shaft':
                draw_angle_visual(img, (int(row['WristX']), int(row['WristY'])), (int(row['TargetX']), int(row['TargetY'])), 
                                  compute_relative_angle((row['WristX'], row['WristY']), (row['TargetX'], row['TargetY']), (100, 500), (400, 500)), 
                                  (100, 500), (400, 500), (0, 255, 0), "Shaft")
            
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
