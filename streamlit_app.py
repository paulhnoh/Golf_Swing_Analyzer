import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="P1-P13 Golf Swing Analyzer", layout="wide")

st.title("⛳ 골프 스윙 P1~P13 정밀 분석 및 미세조정 시스템")
st.markdown("각 페이즈(P1~P13)의 프레임을 슬라이더로 미세조정하면 하단의 분석 테이블에 즉각 반영됩니다.")

# ---------------------------------------------------------
# 1. P1 ~ P13 페이즈 정의 및 초기 프레임 설정
# ---------------------------------------------------------
phases_info = [
    {"phase": "P1", "name": "Address", "desc": "스윙 시작 전 정지 상태", "default": 27},
    {"phase": "P2", "name": "Start Sweep", "desc": "샤프트가 지면과 45°", "default": 38},
    {"phase": "P3", "name": "Back Alignment (Toe Up)", "desc": "샤프트가 지면에 평행", "default": 41},
    {"phase": "P4", "name": "Start Shoulder Back", "desc": "왼팔이 지면에 평행", "default": 116},
    {"phase": "P5", "name": "Backswing Top", "desc": "왼손 최고점 (체공시간 측정)", "default": 163},
    {"phase": "P6", "name": "Transition", "desc": "샤프트가 지면에 135도", "default": 196},
    {"phase": "P7", "name": "DB Alignment (Toe Up)", "desc": "샤프트가 지면에 평행", "default": 197},
    {"phase": "P8", "name": "Impact", "desc": "볼을 타격하는 지점", "default": 198},
    {"phase": "P9", "name": "Lowest Club Head", "desc": "샤프트가 지면에 315도", "default": 205},
    {"phase": "P10", "name": "DF Alignment (Toe Up)", "desc": "샤프트가 지면에 평행", "default": 212},
    {"phase": "P11", "name": "Start Shoulder Forward", "desc": "오른팔이 지면에 평행", "default": 215},
    {"phase": "P12", "name": "Downswing Top", "desc": "오른손 최고점 (체공시간 측정)", "default": 221},
    {"phase": "P13", "name": "Finish", "desc": "스윙이 끝날 때의 정지 상태", "default": 222},
]

# ---------------------------------------------------------
# 2. P5(왼손) / P12(오른손) 최고점 체공시간 계산 로직
# ---------------------------------------------------------
def calculate_peak_duration(y_coords, fps=30, threshold=5.0):
    """
    손의 Y좌표 배열을 받아 최고점(가장 낮은 Y값)에 머문 시간을 초(s) 단위로 계산합니다.
    """
    if not y_coords: return 0.0
    peak_y = min(y_coords)
    # 최고점 기준 오차 범위(threshold) 내에 있는 프레임 수 계산
    peak_frames = [y for y in y_coords if abs(y - peak_y) <= threshold]
    duration = len(peak_frames) / fps
    return round(duration, 3)

# ---------------------------------------------------------
# 3. UI 렌더링: 4열 그리드로 슬라이더 및 스틸컷 배치
# ---------------------------------------------------------
total_frames = 239  # 예시 총 프레임 수
selected_frames = {}

st.subheader("📸 페이즈별 프레임 미세조정")
cols = st.columns(4)

for i, p in enumerate(phases_info):
    with cols[i % 4]:
        # 미세조정 슬라이더
        selected_frames[p['phase']] = st.slider(
            f"[{p['phase']}] 프레임 조정", 
            min_value=0, 
            max_value=total_frames, 
            value=p['default'], 
            key=f"slider_{p['phase']}"
        )
        
        # 💡 실제 영상 연동 시 아래 부분에 cv2로 프레임을 추출하여 표시하는 로직이 들어갑니다.
        # st.image(extracted_frame, use_column_width=True)
        
        # 임시 이미지 플레이스홀더 렌더링
        placeholder_img = np.ones((200, 300, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder_img, f"{p['phase']} Image", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        st.image(placeholder_img, caption=f"[{p['phase']}] Frame: {selected_frames[p['phase']]} / {total_frames}")

st.divider()

# ---------------------------------------------------------
# 4. 분석 데이터 테이블 생성 및 업데이트
# ---------------------------------------------------------
st.subheader("📊 실시간 정밀 분석 데이터 (CSV)")

# 첨부해주신 이미지와 동일한 컬럼 구조
columns = [
    "Phase", "Name", "기준", "Time Stamp(s)", "Frame #", 
    "ShoulderTilt", "Shoulder Rotation", "HipTilt", "Hip Rotation", 
    "LtElbow", "RtElbow", "LtShoulderAngle", "RtShoulderAngle", 
    "LtKnee", "RtKnee", "ClubAngle", "ClubSpeed", "HeadStill Time"
]

data = []
fps = 30 # 기준 FPS

for p in phases_info:
    phase_id = p['phase']
    frame_num = selected_frames[phase_id]
    time_stamp = round(frame_num / fps, 2)
    
    # 💡 P5와 P12의 경우 Pose 모델에서 추출된 손의 Y좌표 이력을 넘겨받아 체공시간을 산출합니다.
    # 아래는 더미 데이터를 활용한 예시입니다.
    head_still_time = ""
    if phase_id == "P5":
        dummy_left_hand_y = [150, 148, 146, 145, 145, 146, 150] # Pose 좌표 예시
        head_still_time = calculate_peak_duration(dummy_left_hand_y, fps)
    elif phase_id == "P12":
        dummy_right_hand_y = [120, 115, 112, 112, 112, 114, 120] # Pose 좌표 예시
        head_still_time = calculate_peak_duration(dummy_right_hand_y, fps)
    
    # 표에 들어갈 한 줄의 데이터 구성
    row = {
        "Phase": phase_id,
        "Name": p['name'],
        "기준": p['desc'],
        "Time Stamp(s)": time_stamp,
        "Frame #": frame_num,
        "ShoulderTilt": 0.0,        # Pose 모델 각도 연산값 바인딩 위치
        "Shoulder Rotation": 0.0,   
        "HipTilt": 0.0,             
        "Hip Rotation": 0.0,        
        "LtElbow": 0.0,             
        "RtElbow": 0.0,             
        "LtShoulderAngle": 0.0,     
        "RtShoulderAngle": 0.0,     
        "LtKnee": 0.0,              
        "RtKnee": 0.0,              
        "ClubAngle": 0.0,           # Custom 모델 샤프트 연산값 바인딩 위치
        "ClubSpeed": 0.0,           
        "HeadStill Time": head_still_time
    }
    data.append(row)

# DataFrame 생성 및 Streamlit 출력
df = pd.DataFrame(data, columns=columns)
st.dataframe(df, use_container_width=True, hide_index=True)

# CSV 다운로드 버튼
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 분석 결과 CSV 다운로드",
    data=csv_data,
    file_name='calibrated_swing_P1_P13.csv',
    mime='text/csv',
)
