import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import os
import tempfile
import shutil
from PIL import Image
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="P1-P13 Auto Golf Swing Analyzer", layout="wide")

st.title("⛳ 골프 스윙 P1~P13 자동 추출 및 정밀 분석 시스템")
st.markdown("영상을 업로드하면 AI가 **모든 프레임을 개별 이미지로 안전하게 추출한 뒤, 한 장씩 정밀 스캔하여 P1~P13 싯점을 찾아냅니다.**")

# ---------------------------------------------------------
# 1. AI 모델 로드
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    pose_model = YOLO('yolov8n-pose.pt') 
    custom_model = YOLO('custom_golf.pt') 
    return pose_model, custom_model

pose_model, custom_model = load_models()

# ---------------------------------------------------------
# 2. 페이즈 기준 정의 및 헬퍼 함수
# ---------------------------------------------------------
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
    peak_frames = [y for y in valid_y if abs(y - peak_y) <= threshold]
    return round(len(peak_frames) / fps, 3)

def find_closest_frame(arr, target, start_idx, end_idx):
    if start_idx >= end_idx or start_idx >= len(arr): return start_idx
    sub_arr = arr[start_idx:end_idx]
    valid_indices = np.where(~np.isnan(sub_arr))[0]
    if len(valid_indices) == 0: return start_idx + (end_idx - start_idx)//2
    closest_sub_idx = valid_indices[np.argmin(np.abs(np.array(sub_arr)[valid_indices] - target))]
    return start_idx + closest_sub_idx

# ---------------------------------------------------------
# 3. 100% 안전한 [이미지 물리적 추출 ➡️ 개별 분석] 엔진
# ---------------------------------------------------------
uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # 세션 초기화 로직 (새 파일 업로드 시)
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name

    if 'auto_frames' not in st.session_state:
        # [1단계] 영상의 모든 프레임을 물리적 이미지 파일(jpg)로 추출
        with st.spinner("1단계: 영상의 모든 프레임을 개별 이미지로 분리하여 서버에 저장 중입니다..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            # 클라우드에 임시 이미지 폴더 생성
            frame_dir = tempfile.mkdtemp()
            st.session_state.frame_dir = frame_dir
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or total_frames > 600: # 최대 20초 분량 제한
                    break
                # 프레임을 명확한 번호의 jpg 파일로 저장 (예: frame_0001.jpg)
                img_path = os.path.join(frame_dir, f"frame_{total_frames:04d}.jpg")
                cv2.imwrite(img_path, frame)
                total_frames += 1
            cap.release()
            st.session_state.total_frames = total_frames
            st.session_state.fps = fps

        # [2단계] 저장된 이미지 파일들을 하나씩 불러와서 정밀 분석
        with st.spinner(f"2단계: 저장된 {total_frames}장의 이미지를 AI가 전수 스캔 중입니다..."):
            left_hand_ys, right_hand_ys = [], []
            left_arm_angles, right_arm_angles, shaft_angles = [], [], []
            
            for f_idx in range(total_frames):
                # 메모리상의 영상이 아닌, 확실히 저장된 이미지 파일을 읽어옴
                img_path = os.path.join(st.session_state.frame_dir, f"frame_{f_idx:04d}.jpg")
                frame = cv2.imread(img_path)
                
                p_res = pose_model(frame, verbose=False)
                c_res = custom_model(frame, verbose=False)
                
                ly, ry, la_angle, ra_angle, s_angle = np.nan, np.nan, np.nan, np.nan, np.nan
                wrist_pt = None
                
                # 손목 및 팔 각도 분석
                if p_res[0].keypoints is not None and len(p_res[0].keypoints.xy) > 0:
                    kpts = p_res[0].keypoints.xy[0].cpu().numpy()
                    if len(kpts) > 10:
                        l_s, r_s = kpts[5], kpts[6] 
                        l_w, r_w = kpts[9], kpts[10] 
                        
                        if l_w[0] > 0: ly = l_w[1]
                        if r_w[0] > 0: ry = r_w[1]
                        
                        if l_s[0] > 0 and l_w[0] > 0:
                            la_angle = abs(math.degrees(math.atan2(l_w[1] - l_s[1], l_w[0] - l_s[0])))
                        if r_s[0] > 0 and r_w[0] > 0:
                            ra_angle = abs(math.degrees(math.atan2(r_w[1] - r_s[1], r_w[0] - r_s[0])))
                            
                        if l_w[0] > 0 and r_w[0] > 0:
                            wrist_pt = ((l_w[0]+r_w[0])/2, (l_w[1]+r_w[1])/2)
                
                # 샤프트(블러) 분석
                head_pt, shaft_pt = None, None
                for box in c_res[0].boxes:
                    cls_name = c_res[0].names[int(box.cls[0].item())]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_pt = ((x1+x2)/2, (y1+y2)/2)
                    
                    if cls_name == 'head': head_pt = center_pt
                    elif cls_name == 'shaft': shaft_pt = center_pt
                
                target_pt = head_pt if head_pt else shaft_pt
                        
                if wrist_pt and target_pt:
                    dx = target_pt[0] - wrist_pt[0]
                    dy = target_pt[1] - wrist_pt[1]
                    s_angle = abs(math.degrees(math.atan2(dy, dx)))
                    
                left_hand_ys.append(ly)
                right_hand_ys.append(ry)
                left_arm_angles.append(la_angle)
                right_arm_angles.append(ra_angle)
                shaft_angles.append(s_angle)

            # 자동 프레임 매핑 로직
            auto_f = {}
            valid_wrist = [y for y in left_hand_ys if not np.isnan(y)]
            p8_idx = left_hand_ys.index(max(valid_wrist)) if valid_wrist else total_frames // 2
            auto_f["P8"] = p8_idx
            auto_f["P1"] = 0 
            
            sub_ly = left_hand_ys[:p8_idx]
            p5_idx = left_hand_ys.index(min([y for y in sub_ly if not np.isnan(y)])) if [y for y in sub_ly if not np.isnan(y)] else p8_idx // 2
            auto_f["P5"] = p5_idx
            
            sub_ry = right_hand_ys[p8_idx:]
            p12_idx = p8_idx + right_hand_ys[p8_idx:].index(min([y for y in sub_ry if not np.isnan(y)])) if [y for y in sub_ry if not np.isnan(y)] else total_frames - 1
            auto_f["P12"] = p12_idx
            
            auto_f["P13"] = total_frames - 1

            auto_f["P2"] = find_closest_frame(shaft_angles, 45, auto_f["P1"], auto_f["P5"])
            auto_f["P3"] = find_closest_frame(shaft_angles, 90, auto_f["P2"], auto_f["P5"])
            auto_f["P4"] = find_closest_frame(left_arm_angles, 0, auto_f["P3"], auto_f["P5"]) 
            
            auto_f["P6"] = find_closest_frame(shaft_angles, 45, auto_f["P5"], auto_f["P8"]) 
            auto_f["P7"] = find_closest_frame(shaft_angles, 90, auto_f["P6"], auto_f["P8"]) 
            
            auto_f["P9"] = find_closest_frame(shaft_angles, 45, auto_f["P8"], auto_f["P12"]) 
            auto_f["P10"] = find_closest_frame(shaft_angles, 90, auto_f["P9"], auto_f["P12"]) 
            auto_f["P11"] = find_closest_frame(right_arm_angles, 0, auto_f["P10"], auto_f["P12"]) 

            st.session_state.p5_time = calculate_peak_duration(left_hand_ys[:p8_idx], fps)
            st.session_state.p12_time = calculate_peak_duration(right_hand_ys[p8_idx:], fps)
            st.session_state.auto_frames = auto_f
            st.session_state.scan_done = True

    # ---------------------------------------------------------
    # 4. 미세조정 UI (물리적 이미지 파일 불러오기)
    # ---------------------------------------------------------
    if 'scan_done' in st.session_state:
        st.subheader("📸 자동 추출 프레임 확인 및 수동 클릭 보정")
        cols = st.columns(4)
        analysis_data = []

        total_frames = st.session_state.total_frames
        fps = st.session_state.fps

        for i, p in enumerate(phases_info):
            with cols[i % 4]:
                phase_id = p['phase']
                auto_frame_num = st.session_state.auto_frames[phase_id]
                
                # 슬라이더
                frame_num = st.slider(f"[{phase_id}] 프레임 조정", 0, total_frames-1, auto_frame_num, key=f"slider_{phase_id}")
                
                # 💡 안전성 핵심: 비디오에서 추출하는 것이 아니라, 이미 저장된 jpg 파일을 확정적으로 불러옵니다.
                img_path = os.path.join(st.session_state.frame_dir, f"frame_{frame_num:04d}.jpg")
                frame = cv2.imread(img_path)
                
                if frame is not None:
                    p_res = pose_model(frame, verbose=False)
                    wrist_pt = None
                    if p_res[0].keypoints is not None and len(p_res[0].keypoints.xy) > 0:
                        kpts = p_res[0].keypoints.xy[0].cpu().numpy()
                        if len(kpts) > 10:
                            lw, rw = kpts[9], kpts[10]
                            if lw[0] > 0 and rw[0] > 0:
                                wrist_pt = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))
                    
                    session_key = f"manual_pt_{phase_id}"
                    if session_key in st.session_state:
                        target_pt = st.session_state[session_key]
                    else:
                        c_res = custom_model(frame, verbose=False)
                        head_pt, shaft_pt = None, None
                        for box in c_res[0].boxes:
                            cls_name = c_res[0].names[int(box.cls[0].item())]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_pt = (int((x1+x2)/2), int((y1+y2)/2))
                            if cls_name == 'head': head_pt = center_pt
                            elif cls_name == 'shaft': shaft_pt = center_pt
                        target_pt = head_pt if head_pt else shaft_pt

                    if wrist_pt and target_pt:
                        cv2.circle(frame, wrist_pt, 8, (0, 255, 255), -1)
                        cv2.circle(frame, target_pt, 8, (0, 0, 255), -1)
                        cv2.line(frame, wrist_pt, target_pt, (0, 255, 0), 4)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    value = streamlit_image_coordinates(Image.fromarray(frame_rgb), key=f"img_{phase_id}")
                    if value is not None:
                        clicked_pt = (value['x'], value['y'])
                        if st.session_state.get(session_key) != clicked_pt:
                            st.session_state[session_key] = clicked_pt
                            st.rerun()
                
                head_still = ""
                if phase_id == "P5": head_still = st.session_state.p5_time
                if phase_id == "P12": head_still = st.session_state.p12_time
                
                row = {
                    "Phase": phase_id, "Name": p['name'], "기준": p['desc'],
                    "Time Stamp(s)": round(frame_num / fps, 2), "Frame #": frame_num,
                    "ShoulderTilt": "", "Shoulder Rotation": "", "HipTilt": "", "Hip Rotation": "", 
                    "LtElbow": "", "RtElbow": "", "LtShoulderAngle": "", "RtShoulderAngle": "", 
                    "LtKnee": "", "RtKnee": "", 
                    "ClubAngle": p['ref_angle'], 
                    "ClubSpeed": "", "HeadStill Time": head_still
                }
                analysis_data.append(row)

        st.divider()
        st.subheader("📊 정밀 분석 데이터 테이블 (자동 추출 결과)")
        
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드", data=csv_data,
            file_name='calibrated_swing_P1_P13.csv', mime='text/csv',
        )
