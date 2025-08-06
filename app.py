# Streamlit 기반 AI 작품 평가 웹앱
# 이미지 업로드 → 밝기, 색상 다양성, 대비 분석 후 점수와 피드백 제공

import streamlit as st
from PIL import Image
import numpy as np
import colorsys

st.set_page_config(page_title="AI 미술 작품 평가기", layout="centered")
st.title("🎨 AI 기반 미술 작품 평가")
st.markdown("작품 사진을 업로드하면, AI가 시각 요소를 분석해 평가해줍니다.")

# ===== 이미지 분석 함수 =====
def analyze_image(image):
    image = image.convert('RGB')
    image_np = np.array(image)

    # 평균 밝기
    brightness = np.mean(image_np)

    # 색상 다양성 (HSV 고유 색상 수)
    hsv_image = image.convert('HSV')
    hsv_np = np.array(hsv_image)
    unique_hues = len(np.unique(hsv_np[:, :, 0]))

    # 대비 (표준편차)
    contrast = np.std(image_np)

    # 점수 계산
    score = (
        (brightness / 255 * 30) +
        (min(unique_hues, 100) / 100 * 30) +
        (min(contrast, 127) / 127 * 40)
    )

    # 피드백
    feedback = []
    if brightness < 80:
        feedback.append("🔅 작품이 다소 어두운 편입니다. 명암 조절을 고려해보세요.")
    if unique_hues < 20:
        feedback.append("🌈 사용된 색상이 적어 보입니다. 다양한 색을 시도해보세요.")
    if contrast < 30:
        feedback.append("⚠️ 대비가 약해 작품이 평면적으로 느껴질 수 있습니다.")

    return round(score, 1), feedback

# ===== 업로드 인터페이스 =====
uploaded_file = st.file_uploader("작품 사진 업로드 (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 작품', use_column_width=True)

    score, feedback = analyze_image(image)

    st.subheader(f"✅ 총점: {score}점 (100점 만점)")
    st.markdown("---")
    st.subheader("📝 AI 피드백")
    if feedback:
        for f in feedback:
            st.markdown(f"- {f}")
    else:
        st.markdown("- 작품의 시각적 요소가 균형 잡혀 있습니다. 멋진 작품이에요! 🎉")
else:
    st.info("왼쪽 사이드바에서 작품 이미지를 업로드해 주세요.")
