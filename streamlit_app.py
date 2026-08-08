import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import math
import cv2
import av
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from yt_dlp import YoutubeDL

st.set_page_config(page_title="AI 정밀 골프 스윙 분석 시스템", layout="wide")

st.title("⛳ AI 정밀 골프 스윙 분석 시스템 (YouTube & Cloud 통합 에디션)")
st.write("PC 리소스 걱정 없이, 내 컴퓨터의 영상 파일을 업로드하거나 유튜브 링크를 입력하여 클라우드 서버에서 즉시 전수 스캔 및 미세조정을 수행합니다.")

# 입력 소스 선택 탭 (1. 파일 업로드 vs 2. 유튜브 URL 자동 수집)
input_method = st.radio("분석할 영상 소스를 선택하세요:", ["📁 내 컴퓨터에서 영상 파일 업로드", "🔗 유튜브(YouTube) 링크로 분석하기"])

video_path = None

if input_method == "📁 내 컴퓨터에서 영상 파일 업로드":
    uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

else:
    yt_url = st.text_input("분석할 유튜브(YouTube) 영상 링크를 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")
    if yt_url:
        if st.button("유튜브 영상 클라우드로 다운로드", type="secondary"):
            with st.spinner("클라우드 서버에서 유튜브 영상을 안전하게 다운로드하고 있습니다..."):
                try:
                    output_dir = tempfile.mkdtemp()
                    out_template = os.path.join(output_dir, 'yt_swing.mp4')
                    
                    ydl_opts = {
                        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
                        'outtmpl': out_template,
                        'noplaylist': True,
                    }
                    with YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    
                    video_path = out_template
                    st.session_state.yt_downloaded_path = video_path
                    st.success("✅ 유튜브 영상 다운로드 및 클라우드 적재 완료!")
                except Exception as e:
                    st.error(f"❌ 유튜브 다운로드 실패: {e}")
        elif 'yt_downloaded_path' in st.session_state:
            video_path = st.session_state.yt_downloaded_path

# 상태 유지 초기화
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'user_frames' not in st.session_state: st.session_state.user_frames = {}

if video_path is not None:
    st.video(video_path)

    if st.button("스윙 정밀 분석 시작 (전수 스캔)", type="primary"):
        with st.spinner("클라우드 서버에서 무손실 프레임 추출 및 관절 스캔을 진행 중입니다..."):
            
            # --- 1. PyAV 무손실 프레임 추출 ---
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
            
            # --- 2. MediaPipe 전수 데이터 스캔 ---
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
            
            landmarks_data = []
            wrist_ys, wrist_xs = [], []
            ls_ys, lw_ys, rs_ys, rw_ys = [], [], [], []
            re_xs, rw_xs = [], []
            hip_ys = []
            
            for frame in frames_bgr:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                h, w, _ = frame.shape
                
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    landmarks_data.append(lm)
                    
                    lw_x, lw_y = lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h
                    rw_x, rw_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h
                    ls_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                    rs_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
                    re_x = lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w
                    rh_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
                    lh_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
                    
                    hx, hy = (lw_x + rw_x) / 2, (lw_y + rw_y) / 2
                    
                    wrist_ys.append(hy); wrist_xs.append(hx)
                    lw_ys.append(lw_y); ls_ys.append(ls_y)
                    rw_ys.append(rw_y); rs_ys.append(rs_y)
                    rw_xs.append(rw_x); re_xs.append(re_x)
                    hip_ys.append((rh_y + lh_y) / 2)
                else:
                    landmarks_data.append(None)
                    for arr in [wrist_ys, wrist_xs, lw_ys, ls_ys, rw_ys, rs_ys, rw_xs, re_xs, hip_ys]:
                        arr.append(np.nan)
            
            st.session_state.lm_data = landmarks_data
            
            # --- 3. 데이터 보간 및 안전 장치 ---
            def smooth(arr):
                s = pd.Series(arr)
                if s.isna().all(): return np.zeros(len(arr))
                return s.interpolate(limit_direction='both').rolling(3, center=True, min_periods=1).mean().values
            
            w_y = smooth(wrist_ys)
            lw_y, ls_y = smooth(lw_ys), smooth(ls_ys)
            rw_y, rs_y = smooth(rw_ys), smooth(rs_ys)
            rw_x, re_x = smooth(rw_xs), smooth(re_xs)
            h_y = smooth(hip_ys)

            # 앵커 및 초기 페이즈 산출
            f_p5 = int(np.nanargmin(w_y)) if not np.all(np.isnan(w_y)) else int(total_frames * 0.4)
            f_p1 = int(np.nanargmax(w_y[:f_p5])) if f_p5 > 0 and not np.all(np.isnan(w_y[:f_p5])) else 0
            
            post_top = w_y[f_p5:min(total_frames, f_p5 + int(fps * 1.5))]
            f_p8 = f_p5 + int(np.nanargmax(post_top)) if len(post_top) > 0 and not np.all(np.isnan(post_top)) else min(total_frames-1, f_p5+10)
            f_p13 = min(total_frames - 1, f_p8 + int(fps * 1.0))

            # 비선형 골든 비율 초기값 배정
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

# --- 4. 반응형 렌더링 및 Zoom 미세 조정 ---
if st.session_state.get('analyzed'):
    st.success("✅ 분석 완료! 클라우드 리소스로 구동 중이며, 슬라이더 조작 시 테이블 수치가 즉각 연동됩니다.")
    
    table_placeholder = st.empty()
    csv_placeholder = st.empty()
    
    st.subheader("📸 단계별 프레임 미세 조정 및 줌(Zoom) 뷰")
    zoom_mode = st.toggle("🔍 2배 확대(Zoom) 모드 켜기 (손목 및 샤프트 중심)", value=False)
    
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
                elif p_code == "P13": draw_shaft("0°", 0)
                
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
    csv_placeholder.download_button("💾 현재 설정된 프레임 데이터 CSV 다운로드", data=csv, file_name='youtube_cloud_analysis.csv', mime='text/csv', type='primary')
