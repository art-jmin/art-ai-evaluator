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
        feedback.append("🔅 작품이 전체적으로 어두운 편이에요. 밝기를 조절하면 시각적 집중도를 높일 수 있어요. 명암 대비를 통해 주제 표현을 더 강하게 해보세요.")
    else:
        feedback.append("🌟 밝기의 활용이 좋습니다. 시선을 자연스럽게 이끌고 있어요.")

    if unique_hues < 20:
        feedback.append("🌈 사용된 색상의 수가 적습니다. 다양한 색조를 활용하면 표현의 폭이 더 넓어지고 시각적 흥미도 증가할 수 있어요.")
    elif unique_hues < 50:
        feedback.append("🖌️ 색상이 적절하게 사용되었어요. 조금 더 다양한 색을 시도해봐도 좋을 것 같아요.")
    else:
        feedback.append("🎨 색채 구성이 매우 풍부합니다. 색상의 조화와 다양성이 돋보여요.")

    if contrast < 30:
        feedback.append("⚠️ 대비가 약해 작품이 다소 평면적으로 보일 수 있어요. 주요 요소와 배경의 대비를 높이면 표현력이 좋아집니다.")
    elif contrast < 60:
        feedback.append("✨ 적절한 대비가 느껴져요. 조금 더 강조가 필요한 부분은 색상 톤이나 밝기를 조절해보세요.")
    else:
        feedback.append("🖼️ 강한 대비 덕분에 작품의 주요 요소가 명확하게 드러나고 있어요. 시각적 완성도가 높습니다.")

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
