import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import os
import tempfile
from PIL import Image
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="P1-P13 Auto Golf Swing Analyzer", layout="wide")
st.title("⛳ 골프 스윙 P1~P13 자동 추출 및 정밀 분석 시스템")

@st.cache_resource
def load_models():
    return YOLO('yolov8n-pose.pt'), YOLO('custom_golf.pt')

pose_model, custom_model = load_models()

# 기준 정의
phases_info = [
    {"phase": "P1", "name": "Address", "desc": "스윙 시작 전 정지 상태", "ref_angle": 0},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트가 지면과 45°", "ref_angle": 45},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트가 지면에 평행", "ref_angle": 90},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔이 지면에 평행", "ref_angle": None},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점", "ref_angle": None},
    {"phase": "P6", "name": "Transition", "desc": "샤프트가 지면과 45°", "ref_angle": 135},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트가 지면에 평행", "ref_angle": 90},
    {"phase": "P8", "name": "Impact", "desc": "볼을 타격하는 지점", "ref_angle": None},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트가 지면과 45°", "ref_angle": 315},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트가 지면에 평행", "ref_angle": 270},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔이 지면에 평행", "ref_angle": None},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점", "ref_angle": None},
    {"phase": "P13", "name": "Finish", "desc": "스윙이 끝날 때의 정지 상태", "ref_angle": None},
]

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'processed' not in st.session_state:
        with st.spinner("이미지 분리 및 AI 전수 스캔 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            y_left, y_right, ang_shaft = [], [], []
            valid_indices = []
            
            for f in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                cv2.imwrite(os.path.join(frame_dir, f"frame_{f:04d}.jpg"), frame)
                
                p = pose_model(frame, verbose=False)[0]
                c = custom_model(frame, verbose=False)[0]
                ly, ry, ang = np.nan, np.nan, np.nan
                
                if p.keypoints is not None and len(p.keypoints.xy[0]) > 10:
                    k = p.keypoints.xy[0].cpu().numpy()
                    if k[5][0] > 0 or k[6][0] > 0:
                        valid_indices.append(f)
                        if k[9][0] > 0: ly = k[9][1]
                        if k[10][0] > 0: ry = k[10][1]
                        
                        head, shaft = None, None
                        for b in c.boxes:
                            name = c.names[int(b.cls[0])]
                            cent = ((b.xyxy[0][0]+b.xyxy[0][2])/2, (b.xyxy[0][1]+b.xyxy[0][3])/2)
                            if name == 'head': head = cent
                            elif name == 'shaft': shaft = cent
                        target = head if head else shaft
                        wrist = ((k[9][0]+k[10][0])/2, (k[9][1]+k[10][1])/2)
                        if target: ang = abs(math.degrees(math.atan2(target[1]-wrist[1], target[0]-wrist[0])))
                
                y_left.append(ly); y_right.append(ry); ang_shaft.append(ang)
            cap.release()

            # 안전 매핑
            p8 = y_left.index(max([y for y in y_left if not np.isnan(y)])) if any(~np.isnan(y_left)) else total_frames//2
            p5 = y_left.index(min([y for y in y_left[:p8] if not np.isnan(y)])) if any(~np.isnan(y_left[:p8])) else max(0, p8-10)
            p12 = p8 + y_right[p8:].index(min([y for y in y_right[p8:] if not np.isnan(y)])) if any(~np.isnan(y_right[p8:])) else min(total_frames-1, p8+10)
            
            auto_f = {"P1": valid_indices[0], "P5": p5, "P8": p8, "P12": p12, "P13": valid_indices[-1]}
            # 나머지 P도 안전하게 매핑
            auto_f["P2"] = min(range(auto_f["P1"], auto_f["P5"]), key=lambda i: abs(ang_shaft[i]-45) if not np.isnan(ang_shaft[i]) else 999)
            auto_f["P3"] = min(range(auto_f["P2"], auto_f["P5"]), key=lambda i: abs(ang_shaft[i]-90) if not np.isnan(ang_shaft[i]) else 999)
            auto_f["P4"] = auto_f["P5"] - 5
            auto_f["P6"] = min(range(auto_f["P5"], auto_f["P8"]), key=lambda i: abs(ang_shaft[i]-135) if not np.isnan(ang_shaft[i]) else 999)
            auto_f["P7"] = min(range(auto_f["P6"], auto_f["P8"]), key=lambda i: abs(ang_shaft[i]-90) if not np.isnan(ang_shaft[i]) else 999)
            auto_f["P9"] = min(range(auto_f["P8"], auto_f["P12"]), key=lambda i: abs(ang_shaft[i]-45) if not np.isnan(ang_shaft[i]) else 999)
            auto_f["P10"] = min(range(auto_f["P9"], auto_f["P12"]), key=lambda i: abs(ang_shaft[i]-90) if not np.isnan(ang_shaft[i]) else 999)
            auto_f["P11"] = auto_f["P12"] - 5
            
            st.session_state.update({"auto_f": auto_f, "frame_dir": frame_dir, "total": total_frames, "fps": fps, "processed": True})

    if 'processed' in st.session_state:
        cols = st.columns(4)
        data = []
        for i, p in enumerate(phases_info):
            with cols[i%4]:
                fn = st.slider(f"[{p['phase']}] 조정", 0, st.session_state.total-1, st.session_state.auto_f[p['phase']], key=p['phase'])
                img = cv2.imread(os.path.join(st.session_state.frame_dir, f"frame_{fn:04d}.jpg"))
                # 자동 크롭
                p_res = pose_model(img, verbose=False)[0]
                if p_res.keypoints is not None and len(p_res.keypoints.xy[0]) > 0:
                    k = p_res.keypoints.xy[0].cpu().numpy()
                    if k[5][0] > 0 and k[6][0] > 0:
                        cx, cy = int((k[5][0]+k[6][0])/2), int((k[5][1]+k[6][1])/2)
                        img = img[max(0, cy-300):cy+300, max(0, cx-300):cx+300]
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                data.append({"Phase": p['phase'], "Name": p['name'], "Frame #": fn, "ClubAngle": p['ref_angle']})
        st.dataframe(pd.DataFrame(data))
