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
st.markdown("영상을 업로드하면 AI가 골퍼 동작 중심으로 자동 줌인하여 P1~P13을 자동 추출합니다.")

@st.cache_resource
def load_models():
    pose_model = YOLO('yolov8n-pose.pt') 
    custom_model = YOLO('custom_golf.pt') 
    return pose_model, custom_model

pose_model, custom_model = load_models()

phases_info = [
    {"phase": "P1", "name": "Address", "desc": "스윙 시작 전 정지 상태", "ref_angle": "0"},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트가 지면과 45°", "ref_angle": "45"},
    {"phase": "P3", "name": "Back Alignment", "desc": "샤프트가 지면에 평행", "ref_angle": "90"},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔이 지면에 평행", "ref_angle": ""},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (체공시간 측정)", "ref_angle": ""},
    {"phase": "P6", "name": "Transition", "desc": "샤프트가 지면과 45°", "ref_angle": "135"},
    {"phase": "P7", "name": "DB Alignment", "desc": "샤프트가 지면에 평행", "ref_angle": "90"},
    {"phase": "P8", "name": "Impact", "desc": "볼을 타격하는 지점", "ref_angle": ""},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트가 지면과 45°", "ref_angle": "315"},
    {"phase": "P10", "name": "DF Alignment", "desc": "샤프트가 지면에 평행", "ref_angle": "270"},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔이 지면에 평행", "ref_angle": ""},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (체공시간 측정)", "ref_angle": ""},
    {"phase": "P13", "name": "Finish", "desc": "스윙이 끝날 때의 정지 상태", "ref_angle": ""},
]

def calculate_peak_duration(y_coords, fps=30, threshold=10.0):
    valid_y = [y for y in y_coords if not np.isnan(y)]
    if not valid_y: return 0.0
    peak_y = min(valid_y)
    return round(len([y for y in valid_y if abs(y - peak_y) <= threshold]) / fps, 3)

def find_closest_frame(arr, target, start_idx, end_idx):
    if start_idx >= end_idx or start_idx >= len(arr): return start_idx
    sub_arr = arr[start_idx:end_idx]
    valid_indices = np.where(~np.isnan(sub_arr))[0]
    return start_idx + valid_indices[np.argmin(np.abs(np.array(sub_arr)[valid_indices] - target))] if len(valid_indices) > 0 else start_idx

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    if 'auto_frames' not in st.session_state:
        with st.spinner("영상 프레임 추출 및 AI 전수 분석 중..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            left_hand_ys, right_hand_ys, left_arm_angles, right_arm_angles, shaft_angles = [], [], [], [], []
            valid_frame_indices = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or total_frames > 600: break
                cv2.imwrite(os.path.join(frame_dir, f"frame_{total_frames:04d}.jpg"), frame)
                
                p_res = pose_model(frame, verbose=False)
                c_res = custom_model(frame, verbose=False)
                ly, ry, la, ra, sa = np.nan, np.nan, np.nan, np.nan, np.nan
                wrist_pt, is_p = None, False
                
                if p_res[0].keypoints is not None and len(p_res[0].keypoints.xy) > 0:
                    kpts = p_res[0].keypoints.xy[0].cpu().numpy()
                    if len(kpts) > 10 and (kpts[5][0] > 0 or kpts[6][0] > 0):
                        is_p = True
                        valid_frame_indices.append(total_frames)
                        ly, ry = kpts[9][1], kpts[10][1]
                        wrist_pt = ((kpts[9][0]+kpts[10][0])/2, (kpts[9][1]+kpts[10][1])/2)
                        la = abs(math.degrees(math.atan2(kpts[9][1] - kpts[5][1], kpts[9][0] - kpts[5][0])))
                        ra = abs(math.degrees(math.atan2(kpts[10][1] - kpts[6][1], kpts[10][0] - kpts[6][0])))
                
                if is_p:
                    h, s = None, None
                    for box in c_res[0].boxes:
                        name = c_res[0].names[int(box.cls[0].item())]
                        c = ((box.xyxy[0][0]+box.xyxy[0][2])/2, (box.xyxy[0][1]+box.xyxy[0][3])/2)
                        if name == 'head': h = c
                        elif name == 'shaft': s = c
                    t = h if h else s
                    if wrist_pt and t: sa = abs(math.degrees(math.atan2(t[1] - wrist_pt[1], t[0] - wrist_pt[0])))
                
                left_hand_ys.append(ly); right_hand_ys.append(ry); left_arm_angles.append(la); right_arm_angles.append(ra); shaft_angles.append(sa)
                total_frames += 1
            cap.release()
            
            p8_idx = left_hand_ys.index(max([y for y in left_hand_ys if not np.isnan(y)]))
            auto_f = {"P1": 0, "P8": p8_idx, "P5": left_hand_ys.index(min([y for y in left_hand_ys[:p8_idx] if not np.isnan(y)])), "P12": p8_idx + right_hand_ys[p8_idx:].index(min([y for y in right_hand_ys[p8_idx:] if not np.isnan(y)])), "P13": total_frames-1}
            auto_f.update({"P2": find_closest_frame(shaft_angles, 45, auto_f["P1"], auto_f["P5"]), "P3": find_closest_frame(shaft_angles, 90, auto_f["P2"], auto_f["P5"]), "P4": find_closest_frame(left_arm_angles, 0, auto_f["P3"], auto_f["P5"]), "P6": find_closest_frame(shaft_angles, 45, auto_f["P5"], auto_f["P8"]), "P7": find_closest_frame(shaft_angles, 90, auto_f["P6"], auto_f["P8"]), "P9": find_closest_frame(shaft_angles, 45, auto_f["P8"], auto_f["P12"]), "P10": find_closest_frame(shaft_angles, 90, auto_f["P9"], auto_f["P12"]), "P11": find_closest_frame(right_arm_angles, 0, auto_f["P10"], auto_f["P12"])})
            
            st.session_state.p5_time, st.session_state.p12_time = calculate_peak_duration(left_hand_ys[:p8_idx], fps), calculate_peak_duration(right_hand_ys[p8_idx:], fps)
            st.session_state.auto_frames, st.session_state.total_frames, st.session_state.fps, st.session_state.scan_done = auto_f, total_frames, fps, True

    if 'scan_done' in st.session_state:
        cols = st.columns(4)
        analysis_data = []
        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                frame_num = st.slider(f"[{p['phase']}] 조정", 0, st.session_state.total_frames-1, st.session_state.auto_frames[p['phase']], key=p['phase'])
                img = cv2.imread(os.path.join(st.session_state.frame_dir, f"frame_{frame_num:04d}.jpg"))
                
                # 자동 크롭 (골퍼 중심)
                p_res = pose_model(img, verbose=False)
                if p_res[0].keypoints is not None and len(p_res[0].keypoints.xy) > 0:
                    kpts = p_res[0].keypoints.xy[0].cpu().numpy()
                    if kpts[5][0] > 0 and kpts[6][0] > 0:
                        cx, cy = int((kpts[5][0]+kpts[6][0])/2), int((kpts[5][1]+kpts[6][1])/2)
                        img = img[max(0, cy-300):min(img.shape[0], cy+300), max(0, cx-300):min(img.shape[1], cx+300)]
                
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"[{p['phase']}]")
                analysis_data.append({"Phase": p['phase'], "Name": p['name'], "Time Stamp(s)": round(frame_num/st.session_state.fps, 2), "ClubAngle": float(p['ref_angle']) if p['ref_angle'] else None, "HeadStill Time": st.session_state.p5_time if p['phase']=="P5" else (st.session_state.p12_time if p['phase']=="P12" else None)})
        
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
        st.download_button("📥 CSV 다운로드", pd.DataFrame(analysis_data).to_csv(index=False).encode('utf-8-sig'), 'swing_data.csv', 'text/csv')
