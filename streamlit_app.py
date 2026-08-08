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

st.title("⛳ AI 정밀 골프 스윙 분석 시스템 (전수 프레임 스캔 엔진)")
st.write("단순 비율 추정을 완전히 폐기했습니다. 영상의 모든 프레임을 스캔하여 샤프트 각도와 관절의 수평/수직 조건을 정확히 충족하는 프레임만을 족집게처럼 찾아냅니다.")

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=["mp4", "mov", "avi"])

if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'user_frames' not in st.session_state: st.session_state.user_frames = {}

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)

    if st.button("전수 프레임 정밀 분석 시작 (시간 소요됨)", type="primary"):
        with st.spinner("AI가 모든 프레임의 관절 좌표와 샤프트 각도를 1장씩 계산 중입니다... (약 1~2분 소요)"):
            
            # 1. 무손실 프레임 추출
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
            
            # 2. MediaPipe 초기화 및 데이터 저장소
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
            
            landmarks_data = []
            wrist_ys, wrist_xs = [], []
            ls_ys, lw_ys, rs_ys, rw_ys = [], [], [], []
            re_xs, rw_xs = [], []
            shaft_angles = []
            
            # --- 3. 0부터 끝까지 모든 프레임 전수 분석 (핵심 로직) ---
            for i, frame in enumerate(frames_bgr):
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                h, w, _ = frame.shape
                
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    landmarks_data.append(lm)
                    
                    # 관절 좌표 픽셀 변환
                    lw_x, lw_y = lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h
                    rw_x, rw_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h
                    ls_x, ls_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                    rs_x, rs_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
                    re_x = lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w
                    
                    hx, hy = (lw_x + rw_x) / 2, (lw_y + rw_y) / 2
                    
                    wrist_ys.append(hy)
                    wrist_xs.append(hx)
                    lw_ys.append(lw_y); ls_ys.append(ls_y)
                    rw_ys.append(rw_y); rs_ys.append(rs_y)
                    re_xs.append(re_x); rw_xs.append(rw_x)
                    
                    # [OpenCV 샤프트 각도 추적] 손목 주변의 직선을 찾아 각도 계산
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, 50, 150)
                    # 손목 주변만 마스킹 (속도 및 정확도 향상)
                    mask = np.zeros_like(edges)
                    cv2.circle(mask, (int(hx), int(hy)), 150, 255, -1)
                    edges_masked = cv2.bitwise_and(edges, mask)
                    
                    lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=10)
                    
                    best_angle = None
                    min_dist = float('inf')
                    
                    if lines is not None:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            # 손목과 직선 사이의 거리 계산
                            dist = min(math.hypot(x1 - hx, y1 - hy), math.hypot(x2 - hx, y2 - hy))
                            if dist < 50: # 손목에 붙어있는 선만 샤프트로 인정
                                # 수직 하단을 0도로 하는 360도 각도 체계
                                vec_x, vec_y = (x1 - x2, y1 - y2) if y1 < y2 else (x2 - x1, y2 - y1)
                                angle = math.degrees(math.atan2(-vec_x, vec_y)) % 360
                                if dist < min_dist:
                                    min_dist = dist
                                    best_angle = angle
                    shaft_angles.append(best_angle)
                else:
                    landmarks_data.append(None)
                    for arr in [wrist_ys, wrist_xs, lw_ys, ls_ys, rw_ys, rs_ys, re_xs, rw_xs, shaft_angles]:
                        arr.append(np.nan)
            
            st.session_state.lm_data = landmarks_data
            
            # --- 4. 데이터 보간 (모션 블러 극복) ---
            # 샤프트가 블러 처리되어 None인 프레임도 앞뒤 프레임을 통해 각도를 유추해냅니다.
            sin_a = pd.Series([math.sin(math.radians(a)) if not np.isnan(a) else np.nan for a in shaft_angles]).interpolate(limit_direction='both').values
            cos_a = pd.Series([math.cos(math.radians(a)) if not np.isnan(a) else np.nan for a in shaft_angles]).interpolate(limit_direction='both').values
            shaft_smooth = [(math.degrees(math.atan2(s, c)) % 360) for s, c in zip(sin_a, cos_a)]
            
            w_y_smooth = pd.Series(wrist_ys).interpolate().rolling(5, center=True, min_periods=1).mean().values

            # --- 5. 1프레임 단위 조건부 검색 알고리즘 (핵심) ---
            # 조건 1. 앵커 고정 (P5, P8, P1)
            f_p5 = int(np.nanargmin(w_y_smooth)) # 양손 Y 최소 (최고점)
            search_end = min(total_frames, f_p5 + int(fps * 1.5))
            f_p8 = f_p5 + int(np.nanargmax(w_y_smooth[f_p5:search_end])) # 양손 Y 최대 (임팩트 최하점)
            f_p1 = int(np.nanargmax(w_y_smooth[:f_p5])) if f_p5 > 0 else 0
            f_p13 = min(total_frames - 1, f_p8 + int(fps * 1.0))

            # 조건 탐색 유틸리티 함수
            def find_closest_angle(start, end, target_angle, angles_list):
                if start >= end: return start
                sub_angles = angles_list[start:end]
                diffs = [min(abs(a - target_angle), 360 - abs(a - target_angle)) for a in sub_angles]
                return start + np.argmin(diffs)

            def find_closest_diff(start, end, arr1, arr2):
                if start >= end: return start
                diffs = np.abs(np.array(arr1[start:end]) - np.array(arr2[start:end]))
                return start + np.nanargmin(diffs)

            # [P4] 왼팔 수평: 어깨Y와 손목Y의 차이가 최소가 되는 지점 (0도)
            f_p4 = find_closest_diff(f_p1, f_p5, lw_ys, ls_ys)
            
            # [P2, P3] 백스윙 샤프트 각도 추적
            f_p2 = find_closest_angle(f_p1, f_p5, 45, shaft_smooth)
            f_p3 = find_closest_angle(f_p1, f_p5, 90, shaft_smooth)
            
            # [P6, P7] 다운스윙 샤프트 각도 추적
            f_p6 = find_closest_angle(f_p5, f_p8, 135, shaft_smooth)
            f_p7 = find_closest_angle(f_p5, f_p8, 90, shaft_smooth)
            
            # [P9, P10] 팔로우스루 샤프트 각도 추적
            f_p9 = find_closest_angle(f_p8, f_p13, 315, shaft_smooth)
            f_p10 = find_closest_angle(f_p8, f_p13, 270, shaft_smooth)
            
            # [P11] 오른팔 수평: 오른 어깨Y와 오른 손목Y 차이가 최소
            f_p11 = find_closest_diff(f_p8, f_p13, rw_ys, rs_ys)
            
            # [P12] 오른팔 수직: 오른 팔꿈치X와 오른 손목X 차이가 최소
            f_p12 = find_closest_diff(f_p11, f_p13, re_xs, rw_xs)

            # 탐색된 프레임 인덱스 할당
            ai_indices = [f_p1, f_p2, f_p3, f_p4, f_p5, f_p6, f_p7, f_p8, f_p9, f_p10, f_p11, f_p12, f_p13]
            ai_indices = [int(max(0, min(total_frames - 1, idx))) for idx in ai_indices]
            
            st.session_state.ai_frames = ai_indices
            phase_keys = [f"P{i}" for i in range(1, 14)]
            st.session_state.user_frames = {phase_keys[i]: ai_indices[i] for i in range(13)}
            st.session_state.analyzed = True

# --- 렌더링 및 수동 미세 조정 섹션 ---
if st.session_state.get('analyzed'):
    st.success("✅ 전수 프레임 스캔 완료! AI가 실제 관절 각도와 샤프트 궤적을 계산하여 최적의 프레임을 찾아냈습니다.")
    
    table_placeholder = st.empty()
    csv_placeholder = st.empty()
    
    st.subheader("📸 단계별 프레임 확인 및 미세 조정")
    zoom_mode = st.toggle("🔍 2배 확대(Zoom) 모드 켜기", value=False)
    
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
                current_f = st.slider(f"[{p_code}] 프레임 조정", 0, total_frames - 1, st.session_state.user_frames[p_code], key=f"slider_{p_code}")
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
                line_w, length = 8, max(w, h) // 3
                
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
                    ex, ey = hx - math.sin(rad) * length, hy + math.cos(rad) * length
                    draw.line([(hx, hy), (ex, ey)], fill=c_line, width=line_w)
                    draw.text((ex + 10, ey - 20), text, font=font, fill=c_text)
                    return ex, ey
                    
                if p_code == "P1": draw_shaft("0°", 0)
                elif p_code == "P2": 
                    ex, ey = draw_shaft("45°", 45)
                    draw.line([(hx, hy), (hx, ey), (ex, ey)], fill=c_sub, width=4)
                elif p_code == "P3": 
                    ex, ey = draw_shaft("90°", 90)
                    draw.line([(hx, hy), (hx, hy + 200)], fill=c_sub, width=4)
                elif p_code == "P4":
                    draw.line([(ls_x, ls_y), (lw_x, lw_y)], fill=c_line, width=line_w)
                    draw.line([(ls_x, ls_y), (ls_x, ls_y + 150)], fill=c_sub, width=4)
                    draw.text((lw_x, lw_y - 50), "Left Arm Parallel", font=font, fill=c_text)
                elif p_code == "P5": draw.text((hx - 80, hy - 120), "Top", font=font, fill=c_text)
                elif p_code == "P6": draw_shaft("135°", 135)
                elif p_code == "P7": draw_shaft("90°", 90)
                elif p_code == "P8":
                    draw.line([(ls_x, ls_y), (lw_x, lw_y)], fill=c_line, width=line_w)
                    draw.line([(lw_x, lw_y), (lw_x, lw_y + length)], fill=c_line, width=line_w)
                    draw.text((lw_x + 40, lw_y + 80), "Impact", font=font, fill=c_text)
                elif p_code == "P9": 
                    ex, ey = draw_shaft("315° (-45°)", 315)
                    draw.line([(hx, hy), (hx, ey), (ex, ey)], fill=c_sub, width=4)
                elif p_code == "P10": 
                    ex, ey = draw_shaft("270°", 270)
                    draw.line([(hx, hy), (hx, hy + 200)], fill=c_sub, width=4)
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
    csv_placeholder.download_button("💾 현재 설정된 프레임 데이터 CSV 다운로드", data=csv, file_name='full_scan_swing.csv', mime='text/csv', type='primary')
