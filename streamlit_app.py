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

try:
    from ultralytics import YOLO
except ImportError:
    pass

st.set_page_config(page_title="AI 정밀 골프 스윙 분석 시스템", layout="wide")

st.title("⛳ AI 정밀 골프 스윙 분석 시스템 (Pro-Zoom & 에러 방지 완벽판)")
st.write("전문가가 직접 교정한 스윙의 '골든 비율(Golden Ratio)'을 AI 초깃값에 이식하였으며, 돋보기(Zoom) 모드를 통해 픽셀 단위의 미세 조정이 가능합니다.")

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=["mp4", "mov", "avi"])

if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'user_frames' not in st.session_state: st.session_state.user_frames = {}

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("AI 정밀 분석 시작 (골든 비율 적용)", type="primary"):
        with st.spinner("프레임 추출 및 앵커 동기화를 진행 중입니다... (약 1~2분 소요)"):
            
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
            wrist_ys = []
            
            for frame in frames_bgr:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                if res.pose_landmarks:
                    landmarks_data.append(res.pose_landmarks.landmark)
                    wy = (res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y + 
                          res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y) / 2
                    wrist_ys.append(wy)
                else:
                    landmarks_data.append(None)
                    wrist_ys.append(np.nan)
                    
            st.session_state.lm_data = landmarks_data
            w_y_smooth = pd.Series(wrist_ys).interpolate().values

            # --- 앵커 고정 (P1, P5, P8) ---
            f_p5 = int(np.nanargmin(w_y_smooth))
            f_p1 = int(np.nanargmax(w_y_smooth[:f_p5])) if f_p5 > 0 else 0
            
            f_p8 = f_p5 + 10
            try:
                model = YOLO('yolov8n.pt')
                results = model(cv2.cvtColor(frames_bgr[f_p1], cv2.COLOR_BGR2RGB), classes=[32], verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # [Bug Fix] TypeError 완전 차단을 위한 1차원 평탄화 및 예외 처리
                    box = results[0].boxes[0].xyxy[0].cpu().numpy().flatten()
                    if len(box) >= 4:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        h_img, w_img, _ = frames_bgr[0].shape
                        x1, y1 = max(0, x1-15), max(0, y1-15)
                        x2, y2 = min(w_img, x2+15), min(h_img, y2+15)
                        
                        base_roi = cv2.cvtColor(frames_bgr[f_p1][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        diffs = []
                        search_range = min(total_frames, f_p5 + int(fps * 1.5))
                        for i in range(f_p5, search_range):
                            roi = cv2.cvtColor(frames_bgr[i][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            diffs.append(np.sum(cv2.absdiff(base_roi, roi)))
                        if diffs: f_p8 = f_p5 + np.argmax(diffs)
                    else:
                        f_p8 = f_p5 + int(np.nanargmax(w_y_smooth[f_p5:min(total_frames, f_p5 + int(fps))]))
                else:
                    f_p8 = f_p5 + int(np.nanargmax(w_y_smooth[f_p5:min(total_frames, f_p5 + int(fps))]))
            except Exception:
                # 에러 발생 시 시스템 멈춤 없이 Fallback 로직 실행
                f_p8 = f_p5 + int(np.nanargmax(w_y_smooth[f_p5:min(total_frames, f_p5 + int(fps))]))

            f_p13 = min(total_frames - 1, f_p8 + int(fps * 1.0))

            # --- 전문가 교정 데이터 기반 비선형 골든 비율 이식 ---
            f_p2 = f_p1 + int((f_p5 - f_p1) * 0.42)
            f_p3 = f_p1 + int((f_p5 - f_p1) * 0.48)
            f_p4 = f_p1 + int((f_p5 - f_p1) * 0.61)
            
            f_p6 = f_p5 + int((f_p8 - f_p5) * 0.70)
            f_p7 = f_p5 + int((f_p8 - f_p5) * 0.84)
            
            f_p9 = f_p8 + int((f_p13 - f_p8) * 0.05)
            f_p10 = f_p8 + int((f_p13 - f_p8) * 0.09)
            f_p11 = f_p8 + int((f_p13 - f_p8) * 0.17)
            f_p12 = f_p8 + int((f_p13 - f_p8) * 0.37)

            ai_indices = [f_p1, f_p2, f_p3, f_p4, f_p5, f_p6, f_p7, f_p8, f_p9, f_p10, f_p11, f_p12, f_p13]
            ai_indices = [int(max(0, min(total_frames - 1, idx))) for idx in ai_indices]
            
            st.session_state.ai_frames = ai_indices
            phase_keys = [f"P{i}" for i in range(1, 14)]
            st.session_state.user_frames = {phase_keys[i]: ai_indices[i] for i in range(13)}
            st.session_state.analyzed = True

# --- 반응형 렌더링 및 Zoom 미세 조정 ---
if st.session_state.get('analyzed'):
    st.success("✅ 세팅이 완료되었습니다. 테이블 수치는 1프레임 조작 시 즉각 반응합니다.")
    
    table_placeholder = st.empty()
    csv_placeholder = st.empty()
    
    st.subheader("📸 단계별 프레임 미세 조정 및 줌(Zoom) 뷰")
    
    zoom_mode = st.toggle("🔍 2배 확대(Zoom) 모드 켜기 (손목 및 샤프트 중심)", value=False)
    if zoom_mode:
        st.info("🔎 확대 모드가 켜졌습니다. 오버레이 라인이 클럽 샤프트 및 관절과 픽셀 단위로 일치하는지 확인하며 슬라이더를 조정하세요.")
    
    frames_bgr = st.session_state.raw_frames
    landmarks_data = st.session_state.lm_data
    fps = st.session_state.fps
    total_frames = len(frames_bgr)
    
    phase_defs = [
        ("P1", "Address", "정지 상태", 0), ("P2", "Start Sweep", "샤프트 45도", 45),
        ("P3", "Back Alignment", "샤프트 평행", 90), ("P4", "Start Shoulder Back", "왼팔 평행", 0),
        ("P5", "Backswing Top", "헤드 정지", 0), ("P6", "Transition", "샤프트 135도", 135),
        ("P7", "DB Alignment", "샤프트 평행", 90), ("P8", "Impact", "볼 타격", 0),
        ("P9", "Lowest Club Head", "샤프트 315도", 315), ("P10", "DF Alignment", "샤프트 평행", 270),
        ("P11", "Start Shoulder Forward", "오른팔 평행", 0), ("P12", "Downswing Top", "최고점 그립", 0),
        ("P13", "Finish", "스윙 끝 정지 상태", 0)
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
                current_f = st.slider(f"[{p_code}] 프레임 조정", 
                                      0, total_frames - 1, 
                                      st.session_state.user_frames[p_code], 
                                      key=f"slider_{p_code}")
                st.session_state.user_frames[p_code] = current_f
                
                raw_img = frames_bgr[current_f]
                img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                
                pil_img = Image.fromarray(img_rgb)
                draw = ImageDraw.Draw(pil_img)
                lm = landmarks_data[current_f]
                h, w, _ = raw_img.shape
                try: font = ImageFont.load_default(size=30)
                except: font = ImageFont.load_default()
                
                c_line, c_sub, c_text = (255, 30, 30), (0, 255, 255), (255, 255, 0)
                line_w, length = 8, 350
                
                if lm:
                    lw_x, lw_y = lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h
                    rw_x, rw_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h
                    ls_x, ls_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                    rs_x, rs_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
                    re_x, re_y = lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h
                    hx, hy = (lw_x + rw_x) / 2, (lw_y + rw_y) / 2
                else:
                    hx, hy, ls_x, ls_y, rs_x, rs_y, re_x, re_y, lw_x, lw_y, rw_x, rw_y = [w//2]*12

                def draw_shaft(text, rad_deg):
                    rad = math.radians(rad_deg)
                    ex, ey = hx + length * math.sin(rad), hy + length * math.cos(rad)
                    draw.line([(hx, hy), (ex, ey)], fill=c_line, width=line_w)
                    draw.text((ex + 10, ey - 20), text, font=font, fill=c_text)
                    return ex, ey
                    
                if p_code == "P1": draw_shaft("0°", 0)
                elif p_code == "P2": 
                    ex, ey = draw_shaft("45°", 135)
                    draw.line([(hx, hy), (hx, ey), (ex, ey)], fill=c_sub, width=4)
                elif p_code == "P3": draw_shaft("90°", 90)
                elif p_code == "P4":
                    draw.line([(ls_x, ls_y), (lw_x, lw_y)], fill=c_line, width=line_w)
                    draw.line([(ls_x, ls_y), (ls_x, ls_y + 150)], fill=c_sub, width=4)
                    draw.text((lw_x, lw_y - 50), "Left Arm Parallel", font=font, fill=c_text)
                elif p_code == "P5": draw.text((hx - 80, hy - 120), "Top", font=font, fill=c_text)
                elif p_code == "P6": draw_shaft("135°", 45)
                elif p_code == "P7": draw_shaft("90°", 90)
                elif p_code == "P8":
                    draw.line([(ls_x, ls_y), (lw_x, lw_y)], fill=c_line, width=line_w)
                    draw.line([(lw_x, lw_y), (lw_x, lw_y + length)], fill=c_line, width=line_w)
                    draw.text((lw_x + 40, lw_y + 80), "Impact", font=font, fill=c_text)
                elif p_code == "P9": 
                    ex, ey = draw_shaft("315° (-45°)", 315)
                    draw.line([(hx, hy), (hx, ey), (ex, ey)], fill=c_sub, width=4)
                elif p_code == "P10": draw_shaft("270°", -90)
                elif p_code == "P11":
                    draw.line([(rs_x, rs_y), (rw_x, rw_y)], fill=c_line, width=line_w)
                    draw.line([(rs_x, rs_y), (rs_x, rs_y + 150)], fill=c_sub, width=4)
                    draw.text((rw_x - 150, rw_y - 50), "Right Arm Parallel", font=font, fill=c_text)
                elif p_code == "P12":
                    draw.line([(re_x, re_y), (rw_x, rw_y)], fill=c_line, width=line_w)
                    draw.text((rw_x + 40, rw_y - 20), "Right Arm Vertical", font=font, fill=c_text)
                elif p_code == "P13": draw.text((hx - 80, hy - 120), "Finish", font=font, fill=c_text)
                
                if zoom_mode and lm:
                    crop_size = int(min(w, h) * 0.6)
                    left, top = max(0, int(hx - crop_size / 2)), max(0, int(hy - crop_size / 2))
                    right, bottom = min(w, int(hx + crop_size / 2)), min(h, int(hy + crop_size / 2))
                    pil_img = pil_img.crop((left, top, right, bottom))
                
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
    csv_placeholder.download_button("💾 현재 설정된 프레임 데이터 CSV 다운로드", data=csv, file_name='golden_ratio_swing_analysis.csv', mime='text/csv', type='primary')
