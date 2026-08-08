import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import datetime
import os
import math
import cv2
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

st.set_page_config(page_title="AI 정밀 골프 스윙 분석 시스템", layout="wide")

st.title("⛳ AI 정밀 골프 스윙 분석 시스템 (실시간 동적 데이터 연동판)")
st.write("전문가가 슬라이더로 프레임을 미세 조정하면, 해당 프레임의 실제 관절 각도(어깨, 골반, 팔꿈치, 무릎)가 실시간으로 재계산되어 테이블에 즉시 업데이트됩니다.")

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=["mp4", "mov", "avi"])

# 상태 유지 초기화
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'user_frames' not in st.session_state: st.session_state.user_frames = {}

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("AI 정밀 분석 시작", type="primary"):
        with st.spinner("AI가 영상을 분석하고 생체역학 데이터를 추출하고 있습니다..."):
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps): fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
            
            frames_bgr = []
            landmarks_data = []
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames_bgr.append(frame)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                landmarks_data.append(res.pose_landmarks.landmark if res.pose_landmarks else None)
                
            cap.release()
            
            st.session_state.raw_frames = frames_bgr
            st.session_state.lm_data = landmarks_data
            st.session_state.fps = fps
            
            # --- AI 기본 프레임 탐색 로직 (이전과 동일하게 탑/임팩트 등 추적) ---
            wrist_ys = []
            for lm in landmarks_data:
                if lm: wrist_ys.append((lm[mp_pose.PoseLandmark.LEFT_WRIST].y + lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)/2)
                else: wrist_ys.append(np.nan)
            
            w_y = pd.Series(wrist_ys).interpolate().values
            f_p5 = int(np.nanargmin(w_y)) # Top
            search_end = min(len(frames_bgr), f_p5 + int(fps * 1.0))
            f_p8 = f_p5 + int(np.nanargmax(w_y[f_p5:search_end])) # Impact
            
            f_p1 = max(0, f_p5 - int(fps * 1.5))
            f_p13 = min(len(frames_bgr) - 1, f_p8 + int(fps * 1.0))
            
            ai_indices = [
                f_p1, (f_p1+f_p5)//3, (f_p1+f_p5)//2, f_p5 - int(fps*0.2), f_p5, 
                f_p5 + int(fps*0.1), f_p5 + int(fps*0.2), f_p8, f_p8 + int(fps*0.1), 
                f_p8 + int(fps*0.2), f_p8 + int(fps*0.3), f_p8 + int(fps*0.5), f_p13
            ]
            
            st.session_state.ai_frames = ai_indices
            phase_keys = [f"P{i}" for i in range(1, 14)]
            st.session_state.user_frames = {phase_keys[i]: ai_indices[i] for i in range(13)}
            st.session_state.analyzed = True

# --- 분석 완료 후 반응형 렌더링 섹션 ---
if st.session_state.get('analyzed'):
    st.success("✅ 분석 완료! 하단의 프레임을 조정하면 테이블 수치가 즉각적으로 재계산되어 연동됩니다.")
    
    # 1. 동적 테이블이 들어갈 빈 공간(Placeholder) 생성
    table_placeholder = st.empty()
    csv_placeholder = st.empty()
    
    # 2. 미세조정 슬라이더 렌더링
    st.subheader("📸 단계별 프레임 미세 조정 (수동 변경 시 테이블 실시간 업데이트)")
    
    frames_bgr = st.session_state.raw_frames
    landmarks_data = st.session_state.lm_data
    fps = st.session_state.fps
    
    phase_defs = [
        ("P1", "Address", "스윙 시작 전 정지 상태", 0), ("P2", "Start Sweep", "샤프트 45도", 45),
        ("P3", "Back Alignment", "샤프트 평행", 90), ("P4", "Start Shoulder Back", "왼팔 평행", 0),
        ("P5", "Backswing Top", "헤드 정지", 0), ("P6", "Transition", "샤프트 135도", 135),
        ("P7", "DB Alignment", "샤프트 평행", 90), ("P8", "Impact", "볼 타격", 0),
        ("P9", "Lowest Club Head", "샤프트 315도", 315), ("P10", "DF Alignment", "샤프트 평행", 270),
        ("P11", "Start Shoulder Forward", "오른팔 평행", 0), ("P12", "Downswing Top", "최고점 그립", 0),
        ("P13", "Finish", "스윙 끝 정지 상태", 0)
    ]
    
    mp_pose = mp.solutions.pose
    
    # 실제 각도 계산 함수 (좌표 기반)
    def get_angle(lm, p1, p2, p3):
        if not lm: return 0.0
        a, b, c = lm[p1], lm[p2], lm[p3]
        ang = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
        ang = abs(ang)
        return round(ang if ang <= 180 else 360 - ang, 1)

    def get_tilt(lm, p_left, p_right):
        if not lm: return 0.0
        l, r = lm[p_left], lm[p_right]
        return round(math.degrees(math.atan2(r.y - l.y, r.x - l.x)), 1)
        
    full_swing_data = []
    
    # 4열 배치 슬라이더 및 스틸컷 출력
    for row_start in range(0, 13, 4):
        cols = st.columns(4)
        for i in range(4):
            idx = row_start + i
            if idx >= 13: break
            
            p_code, p_name, p_desc, fixed_angle = phase_defs[idx]
            
            with cols[i]:
                # 슬라이더: 값 변경 시 st.session_state 즉각 업데이트됨
                current_f = st.slider(f"[{p_code}] 프레임 조정", 
                                      0, len(frames_bgr)-1, 
                                      st.session_state.user_frames[p_code], 
                                      key=f"slider_{p_code}")
                st.session_state.user_frames[p_code] = current_f
                
                # 이미지 출력
                raw_img = frames_bgr[current_f]
                img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"[{p_code}] Frame: {current_f}", use_container_width=True)
                
                # --- 동적 테이블용 실제 생체 데이터 연산 ---
                lm = landmarks_data[current_f]
                t_stamp = round(current_f / fps, 2)
                
                row = {
                    "Phase": p_code,
                    "Name": p_name,
                    "기준": p_desc,
                    "Timestamp(s)": t_stamp,
                    "Frame #": current_f,
                    # 이전의 가짜 데이터를 버리고 '실제 픽셀 각도'를 실시간 계산!
                    "Shoulder Tilt": get_tilt(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    "Hip Tilt": get_tilt(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                    "LtElbow": get_angle(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    "RtElbow": get_angle(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    "LtKnee": get_angle(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    "RtKnee": get_angle(lm, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    "ClubAngle Ref": fixed_angle
                }
                full_swing_data.append(row)

    # 3. 빈 공간(Placeholder)에 실시간 계산된 테이블 업데이트
    df_result = pd.DataFrame(full_swing_data)
    table_placeholder.dataframe(df_result.set_index("Phase"), use_container_width=True)
    
    # 4. 실시간 CSV 다운로드 연동
    csv = df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    csv_placeholder.download_button(
        label="💾 현재 설정된 프레임 데이터 CSV 다운로드",
        data=csv,
        file_name='dynamic_swing_analysis.csv',
        mime='text/csv',
        type='primary'
    )
