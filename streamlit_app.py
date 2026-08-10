# 1. 필수 라이브러리 설치
!pip install -q streamlit ultralytics opencv-python-headless pillow numpy streamlit-image-coordinates yt-dlp
!npm install -g localtunnel

# 2. 통합 에디션 앱 코드 생성
%%writefile app.py
import streamlit as st
import cv2
import numpy as np
import math
import os
import tempfile
from PIL import Image
from yt_dlp import YoutubeDL
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="AI Golf Swing Analyzer", layout="wide")

# 모델 로드 캐싱
@st.cache_resource
def load_models():
    pose_model = YOLO('yolov8n-pose.pt') 
    custom_model = YOLO('runs/detect/custom_golf/weights/best.pt') 
    return pose_model, custom_model

pose_model, custom_model = load_models()

st.title("⛳ AI 정밀 골프 스윙 분석 시스템 (YouTube & Cloud 통합 에디션)")
st.markdown("PC 리소스 걱정 없이, 내 컴퓨터의 영상 파일을 업로드하거나 유튜브 링크를 입력하여 클라우드 서버에서 즉시 전수 스캔 및 미세조정을 수행합니다.")

# 입력 소스 변경 시 세션 초기화
def reset_state():
    for key in ['impact_img', 'wrist_pt', 'head_pt', 'video_processed']:
        if key in st.session_state:
            del st.session_state[key]

source_type = st.radio("분석할 영상 소스를 선택하세요:", ["내 컴퓨터에서 영상 파일 업로드", "유튜브(YouTube) 링크로 분석하기"], on_change=reset_state)

video_path = None

# UI 렌더링 및 비디오 경로 확보
if source_type == "내 컴퓨터에서 영상 파일 업로드":
    uploaded_file = st.file_uploader("스윙 영상을 업로드하세요 (MP4, MOV 등)", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
else:
    yt_url = st.text_input("유튜브 링크를 입력하세요")
    if yt_url:
        with st.spinner("유튜브 영상을 클라우드로 가져오는 중입니다..."):
            ydl_opts = {'format': 'bestvideo[ext=mp4]/best[ext=mp4]', 'outtmpl': 'temp_yt_video.mp4', 'quiet': True, 'overwrite': True}
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([yt_url])
                video_path = 'temp_yt_video.mp4'
            except Exception as e:
                st.error("유튜브 영상 다운로드에 실패했습니다. 다른 링크를 시도해주세요.")

# 영상 분석 로직
if video_path:
    if st.button("🚀 영상 분석 시작") or 'video_processed' in st.session_state:
        # 최초 1회만 영상을 분석하여 임팩트 프레임을 추출
        if 'video_processed' not in st.session_state:
            with st.spinner("영상에서 임팩트 순간을 스캔 중입니다... (최대 1~2분 소요)"):
                cap = cv2.VideoCapture(video_path)
                frames = []
                wrist_ys = []
                
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_count > 600: # 최대 20초 스캔 (속도 최적화)
                        break
                    
                    if frame_count % 2 == 0:
                        res = pose_model(frame, verbose=False)
                        wy = np.nan
                        lw_pt, rw_pt = None, None
                        if res[0].keypoints is not None and len(res[0].keypoints.xy) > 0:
                            kpts = res[0].keypoints.xy[0].cpu().numpy()
                            if len(kpts) > 10:
                                lw, rw = kpts[9], kpts[10]
                                if lw[0] > 0 and rw[0] > 0:
                                    wy = (lw[1] + rw[1]) / 2
                                    lw_pt, rw_pt = lw, rw
                                else:
                                    wy = max(lw[1], rw[1])
                                    lw_pt, rw_pt = lw, rw
                        wrist_ys.append(wy if wy > 0 else np.nan)
                        frames.append((frame, lw_pt, rw_pt))
                    frame_count += 1
                cap.release()

                if np.isnan(wrist_ys).all():
                    st.error("영상에서 골퍼를 찾을 수 없습니다. 정면 스윙 영상인지 확인해주세요.")
                else:
                    # 임팩트 순간 (손목이 가장 낮게 내려온 프레임) 추출
                    impact_idx = int(np.nanargmax(wrist_ys))
                    impact_frame, lw, rw = frames[impact_idx]
                    
                    if lw[0] > 0 and rw[0] > 0:
                        wrist_pt = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))
                    else:
                        wrist_pt = (int(max(lw[0], rw[0])), int(max(lw[1], rw[1])))
                        
                    # 커스텀 AI 모델로 클럽 헤드 찾기
                    custom_res = custom_model(impact_frame, verbose=False)
                    head_pt = None
                    for box in custom_res[0].boxes:
                        cls_id = int(box.cls[0].item())
                        if custom_res[0].names[cls_id] == 'head':
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            head_pt = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            break
                    
                    # BGR을 웹용 RGB로 변환 후 세션에 저장
                    st.session_state.impact_img = cv2.cvtColor(impact_frame, cv2.COLOR_BGR2RGB)
                    st.session_state.wrist_pt = wrist_pt
                    st.session_state.head_pt = head_pt
                    st.session_state.video_processed = True

        # 분석 완료 후 UI 렌더링
        if 'video_processed' in st.session_state:
            draw_img = st.session_state.impact_img.copy()
            wrist_pt = st.session_state.wrist_pt
            head_pt = st.session_state.head_pt
            shaft_angle = 0.0

            if wrist_pt and head_pt:
                cv2.circle(draw_img, wrist_pt, 8, (0, 255, 255), -1)
                cv2.circle(draw_img, head_pt, 8, (255, 0, 0), -1) # RGB 환경이므로 Red는 (255,0,0)
                cv2.line(draw_img, wrist_pt, head_pt, (0, 255, 0), 4)
                
                dx = head_pt[0] - wrist_pt[0]
                dy = head_pt[1] - wrist_pt[1]
                shaft_angle = abs(math.degrees(math.atan2(dy, dx)))
                cv2.putText(draw_img, f"Shaft Angle: {shaft_angle:.1f} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif wrist_pt and not head_pt:
                cv2.circle(draw_img, wrist_pt, 8, (0, 255, 255), -1)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📸 임팩트 자동 포착 & 수동 보정")
                # 이미지 클릭 UI
                value = streamlit_image_coordinates(Image.fromarray(draw_img), key="pil")
                
                if value is not None:
                    clicked_pt = (value['x'], value['y'])
                    if st.session_state.head_pt != clicked_pt:
                        st.session_state.head_pt = clicked_pt
                        st.rerun() # 선 재계산 및 렌더링

            with col2:
                st.subheader("📊 역추적 데이터")
                if wrist_pt: st.success(f"**손목 좌표:** X:{wrist_pt[0]}, Y:{wrist_pt[1]}")
                
                if head_pt:
                    st.success(f"**헤드 좌표:** X:{head_pt[0]}, Y:{head_pt[1]}")
                    st.info(f"**샤프트 궤적 각도:** {shaft_angle:.1f}°")
                else:
                    st.error("⚠️ AI가 헤드를 놓쳤습니다. 사진 속 헤드 위치를 클릭하세요.")
                    
                st.write("💡 **전문가 미세조정(Expert UI):** AI가 추적한 빨간색 헤드 점의 위치가 미세하게 어긋났다면, 사진 속 실제 헤드 중심을 클릭하세요. 마우스를 따라 선이 즉시 보정됩니다.")

# 3. 터널링 링크 생성 및 앱 구동
import urllib
import time

ip_password = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
print(f"🔑 [중요] 아래 링크 클릭 후 보안 화면에 입력할 IP 비밀번호: {ip_password}\n")

get_ipython().system_raw('streamlit run app.py &>/content/logs.txt &')
time.sleep(3)

!npx localtunnel --port 8501
