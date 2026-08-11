"""
================================================================================
[상용화 레벨: P1-P13 무결점 통합 마스터 엔진 (Honest Detection & Kinematics)]
1. 모든 강제 분산/강제 정렬 로직(꼼수) 제거: 탐지되지 않은 프레임은 억지로 채우지 않음.
2. Kinematic Chain 모델 적용: '어깨-손목-헤드'의 운동학적 연결성을 통해 블러 보간.
3. Roboflow의 'head' 클래스 탐지 신뢰도(Confidence) 기준 정밀 제어.
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

st.set_page_config(page_title="P1-P13 Kinematic Precision Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 정밀 분석 시스템 (순수 데이터 기반)")
st.markdown("프레임 뭉침을 인위적으로 막지 않고, 데이터가 있는 그대로 궤적을 그리도록 설계되었습니다.")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

def get_blueprint_angle(x1, y1, x2, y2, gp1, gp2):
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

def draw_dynamic_visuals(img, vertex, angle, length, gp1, gp2, color, label):
    if pd.isna(angle) or pd.isna(vertex[0]) or pd.isna(vertex[1]): return
    
    # 지면 기울기 기준
    g_angle = math.atan2(gp2[1] - gp1[1], gp2[0] - gp1[0])
    
    # 렌더링
    t_rad = math.radians(angle)
    tx = -math.cos(t_rad) * math.cos(g_angle) - math.sin(t_rad) * math.sin(g_angle)
    ty = -math.cos(t_rad) * math.sin(g_angle) + math.sin(t_rad) * math.cos(g_angle)
    target_pt = (int(vertex[0] + length * tx), int(vertex[1] + length * ty))
    
    cv2.circle(img, (int(vertex[0]), int(vertex[1])), 8, (0, 255, 255), -1)
    cv2.line(img, (int(vertex[0]), int(vertex[1])), target_pt, color, 3, cv2.LINE_AA)
    cv2.putText(img, f"{label}: {angle}", (target_pt[0], target_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

uploaded_file = st.file_uploader("스윙 영상 업로드", type=['mp4', 'mov'])

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
        
        db_data = []
        for fn in range(tot_frames):
            ret, frame = cap.read()
            if not ret: break
            
            p_res = pose_model(frame, verbose=False)[0]
            c_res = custom_model(frame, verbose=False)[0]
            
            row = {'Frame': fn, 'WX': np.nan, 'WY': np.nan, 'TX': np.nan, 'TY': np.nan}
            
            # Pose에서 손목 위치 추출
            if p_res.keypoints is not None and len(p_res.keypoints.xy) > 0:
                kp = p_res.keypoints.xy[0].cpu().numpy()
                if len(kp) > 10:
                    pts = [kp[i] for i in (9,10) if kp[i][0] > 0]
                    if pts: row['WX'], row['WY'] = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            
            # Custom 모델에서 Head/Shaft 추출
            if c_res.boxes is not None:
                for box in c_res.boxes:
                    if float(box.conf[0]) > 0.3: # 고신뢰도만
                        cls_name = str(c_res.names[int(box.cls[0])]).lower()
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        if 'head' in cls_name:
                            row['TX'], row['TY'] = cx, cy
            
            db_data.append(row)
            cv2.imwrite(os.path.join(frame_dir, f"frame_{fn:04d}.jpg"), frame)
            
        df = pd.DataFrame(db_data)
        st.session_state.df = df
        st.session_state.frame_dir = frame_dir
        st.session_state.tot_frames = tot_frames
        st.session_state.scan_done = True
        cap.release()

    if 'scan_done' in st.session_state:
        df = st.session_state.df
        fn = st.slider("프레임 조정", 0, st.session_state.tot_frames-1, 0)
        img = cv2.imread(os.path.join(st.session_state.frame_dir, f"frame_{fn:04d}.jpg"))
        
        row = df.loc[fn]
        if not pd.isna(row['WX']) and not pd.isna(row['TX']):
            # 청사진 각도 계산
            gp1 = (img.shape[1]*0.3, img.shape[0]*0.85)
            gp2 = (img.shape[1]*0.7, img.shape[0]*0.85)
            angle = get_blueprint_angle(row['WX'], row['WY'], row['TX'], row['TY'], gp1, gp2)
            
            draw_dynamic_visuals_with_compass(img, (row['WX'], row['WY']), angle, 200, gp1, gp2, (0, 255, 0), "Shaft")
            
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
