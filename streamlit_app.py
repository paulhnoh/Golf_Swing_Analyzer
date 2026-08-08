import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import datetime
import os
import math
import cv2
import av
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

st.set_page_config(page_title="AI 정밀 골프 스윙 분석 시스템", layout="wide")

st.title("⛳ AI 정밀 골프 스윙 분석 시스템 (좌표계 벡터 완벽 교정판)")
st.write("전문가 수동 프레임 조정 기능은 유지하며, 스윙 방향에 맞춘 완벽한 벡터 계산으로 PDF와 100% 동일한 직각 삼각형과 각도 오버레이를 구현합니다.")

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=["mp4", "mov", "avi"])

if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'user_frames' not in st.session_state: st.session_state.user_frames = {}

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("영상 스캔 및 분석 준비", type="primary"):
        with st.spinner("프레임 추출 및 MediaPipe 관절 좌표를 스캔 중입니다..."):
            container = av.open(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            
            frames_bgr = []
            for frame in container.decode(stream):
                frames_bgr.append(frame.to_ndarray(format='bgr24'))
            container.close()
            
            total_frames = len(frames_bgr)
            st.session_state.raw_frames = frames_bgr
            st.session_state.fps = fps
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
            
            landmarks_data = []
            for frame in frames_bgr:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                landmarks_data.append(res.pose_landmarks.landmark if res.pose_landmarks else None)
                    
            st.session_state.lm_data = landmarks_data
            
            # 수동 조정을 위한 가이드라인 선형 분배 (사용자가 슬라이더로 맞춤)
            f_top = int(total_frames * 0.4)
            f_imp = int(total_frames * 0.7)
            ai_indices = [
                0, int(f_top*0.25), int(f_top*0.5), int(f_top*0.75), f_top,
                f_top + int((f_imp-f_top)*0.33), f_top + int((f_imp-f_top)*0.66), f_imp,
                f_imp + int((total_frames-1-f_imp)*0.2), f_imp + int((total_frames-1-f_imp)*0.4),
                f_imp + int((total_frames-1-f_imp)*0.6), f_imp + int((total_frames-1-f_imp)*0.8), total_frames - 1
            ]
            
            phase_keys = [f"P{i}" for i in range(1, 14)]
            st.session_state.user_frames = {phase_keys[i]: ai_indices[i] for i in range(13)}
            st.session_state.analyzed = True

if st.session_state.get('analyzed'):
    st.success("✅ 영상 스캔 완료! 하단 슬라이더로 프레임을 맞추면 완벽히 교정된 오버레이가 나타납니다.")
    
    table_placeholder = st.empty()
    csv_placeholder = st.empty()
    
    frames_bgr = st.session_state.raw_frames
    landmarks_data = st.session_state.lm_data
    fps = st.session_state.fps
    total_frames = len(frames_bgr)
    
    phase_defs = [
        ("P1", "Address", "정지 상태", 0), ("P2", "Start Sweep", "샤프트 45도", 45),
        ("P3", "Back Alignment", "샤프트 90도", 90), ("P4", "Start Shoulder Back", "왼팔 수평", 0),
        ("P5", "Backswing Top", "헤드 정지", 0), ("P6", "Transition", "샤프트 135도", 135),
        ("P7", "DB Alignment", "샤프트 90도", 90), ("P8", "Impact", "볼 타격", 0),
        ("P9", "Lowest Club Head", "샤프트 315도", 315), ("P10", "DF Alignment", "샤프트 270도", 270),
        ("P11", "Start Shoulder Forward", "오른팔 수평", 0), ("P12", "Downswing Top", "오른팔 수직", 0),
        ("P13", "Finish", "정지 상태", 0)
    ]
    
    mp_pose = mp.solutions.pose
    
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
    
    for row_start in range(0, 13, 4):
        cols = st.columns(4)
        for i in range(4):
            idx = row_start + i
            if idx >= 13: break
            p_code, p_name, p_desc, fixed_angle = phase_defs[idx]
            
            with cols[i]:
                current_f = st.slider(f"[{p_code}] 프레임 조정", 0, total_frames - 1, st.session_state.user_frames[p_code], key=f"slider_{p_code}")
                st.session_state.user_frames[p_code] = current_f
                
                raw_img = frames_bgr[current_f]
                img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                draw = ImageDraw.Draw(pil_img)
                lm = landmarks_data[current_f]
                h, w, _ = raw_img.shape
                
                try: font = ImageFont.load_default(size=25)
                except: font = ImageFont.load_default()
                
                c_line, c_sub, c_text = (255, 30, 30), (0, 255, 255), (255, 255, 0)
                line_w, length = 6, 250
                
                if lm:
                    lw_x, lw_y = lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h
                    rw_x, rw_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h
                    ls_x, ls_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                    rs_x, rs_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
                    re_x, re_y = lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h
                    hx, hy = (lw_x + rw_x) / 2, (lw_y + rw_y) / 2
                else:
                    hx, hy, ls_x, ls_y, rs_x, rs_y, re_x, re_y, lw_x, lw_y, rw_x, rw_y = [w//2]*12

                # [핵심 수정] 골프 스윙에 맞춘 360도 벡터 변환 (0=하단, 90=좌측, 180=상단, 270=우측)
                def draw_shaft(text, angle_deg):
                    rad = math.radians(angle_deg)
                    dx = -math.sin(rad) * length
                    dy = math.cos(rad) * length
                    ex, ey = hx + dx, hy + dy
                    draw.line([(hx, hy), (ex, ey)], fill=c_line, width=line_w)
                    draw.text((ex + 10, ey - 10), text, font=font, fill=c_text)
                    return ex, ey
                    
                if p_code == "P1": 
                    draw_shaft("0°", 0)
                elif p_code == "P2": 
                    ex, ey = draw_shaft("45°", 45)
                    # 수직/수평선으로 완벽한 직각 삼각형 형성
                    draw.line([(hx, hy), (hx, ey)], fill=c_sub, width=3)
                    draw.line([(hx, ey), (ex, ey)], fill=c_sub, width=3)
                elif p_code == "P3": 
                    ex, ey = draw_shaft("90°", 90)
                    draw.line([(hx, hy), (hx, hy + 180)], fill=c_sub, width=3)
                elif p_code == "P4":
                    draw.line([(ls_x, ls_y), (lw_x, lw_y)], fill=c_line, width=line_w)
                    draw.line([(ls_x, ls_y), (ls_x, ls_y + 180)], fill=c_sub, width=3)
                    draw.text((lw_x - 100, lw_y - 30), "Left Arm Parallel", font=font, fill=c_text)
                elif p_code == "P5": 
                    draw.text((hx - 50, hy - 80), "Top", font=font, fill=c_text)
                elif p_code == "P6": 
                    draw_shaft("135°", 135)
                elif p_code == "P7": 
                    draw_shaft("90°", 90)
                elif p_code == "P8":
                    draw.line([(ls_x, ls_y), (lw_x, lw_y)], fill=c_line, width=line_w)
                    draw.line([(lw_x, lw_y), (lw_x, lw_y + length)], fill=c_line, width=line_w)
                    draw.text((lw_x + 20, lw_y + 50), "Impact", font=font, fill=c_text)
                elif p_code == "P9": 
                    ex, ey = draw_shaft("315° (-45°)", 315)
                    draw.line([(hx, hy), (hx, ey)], fill=c_sub, width=3)
                    draw.line([(hx, ey), (ex, ey)], fill=c_sub, width=3)
                elif p_code == "P10": 
                    ex, ey = draw_shaft("270°", 270)
                    draw.line([(hx, hy), (hx, hy + 180)], fill=c_sub, width=3)
                elif p_code == "P11":
                    draw.line([(rs_x, rs_y), (rw_x, rw_y)], fill=c_line, width=line_w)
                    draw.line([(rs_x, rs_y), (rs_x, rs_y + 180)], fill=c_sub, width=3)
                    draw.text((rw_x - 120, rw_y - 30), "Right Arm Parallel", font=font, fill=c_text)
                elif p_code == "P12":
                    draw.line([(re_x, re_y), (rw_x, rw_y)], fill=c_line, width=line_w)
                    draw.text((rw_x + 20, rw_y - 20), "Right Arm Vertical", font=font, fill=c_text)
                elif p_code == "P13": 
                    draw.text((hx - 50, hy - 80), "Finish", font=font, fill=c_text)
                
                st.image(np.array(pil_img), caption=f"[{p_code}] Frame: {current_f} / {total_frames-1}", use_container_width=True)
                
                t_stamp = round(current_f / fps, 2)
                row = {
                    "Phase": p_code, "Name": p_name, "기준": p_desc,
                    "Timestamp(s)": t_stamp, "Frame #": current_f,
                    "Shoulder Tilt": get_tilt(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    "Hip Tilt": get_tilt(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                    "LtElbow": get_angle(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    "RtElbow": get_angle(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    "LtKnee": get_angle(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    "RtKnee": get_angle(lm, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    "ClubAngle Ref": fixed_angle
                }
                full_swing_data.append(row)

    df_result = pd.DataFrame(full_swing_data)
    table_placeholder.dataframe(df_result.set_index("Phase"), use_container_width=True)
    csv = df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    csv_placeholder.download_button("💾 현재 설정된 데이터 CSV 다운로드", data=csv, file_name='calibrated_swing.csv', mime='text/csv', type='primary')
