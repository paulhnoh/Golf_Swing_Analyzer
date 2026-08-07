import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile

st.set_page_config(page_title="AI 골프 스윙 분석 시스템", layout="wide")
st.title("⛳ AI 정밀 골프 스윙 분석 시스템")

st.success("시스템 환경이 성공적으로 로드되었습니다! 영상을 업로드해 주세요.")

uploaded_file = st.file_uploader("스윙 영상을 업로드하세요", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)
    st.info("영상이 정상적으로 업로드되었습니다.")
