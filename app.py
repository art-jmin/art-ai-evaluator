# Streamlit 기반 AI 작품 평가 웹앱 (캐릭터 그림 전용 평가)
# 이미지 업로드 → AI가 그림 여부 판별 → 설명 + 평가 (형태력, 묘사력, 자연스러움, 색상 등)

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision import models

st.set_page_config(page_title="AI 캐릭터 그림 평가기", layout="centered")
st.title("🎨 캐릭터 그림 AI 평가기")
st.markdown("작품 사진을 업로드하면, AI가 그림 여부를 판단하고 평가해줍니다.")

# ===== 그림 vs 사진 구분 모델 설정 (간단 CNN 전이 학습 기반) =====
@st.cache_resource(allow_output_mutation=True)
def load_clip_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def classify_image_type(image):
    # 이 함수는 간단한 예시이므로 실제로는 학습된 모델 사용 필요
    # 임시로 색상 수 기준으로 구분
    image = image.resize((224, 224))
    image_np = np.array(image.convert('RGB'))
    unique_colors = len(np.unique(image_np.reshape(-1, 3), axis=0))
    return "그림" if unique_colors < 15000 else "사진"

# ===== 캐릭터 그림 평가 함수 =====
def evaluate_character_art(image):
    image = image.convert('RGB')
    image_np = np.array(image)

    # 시각 요소 분석
    brightness = np.mean(image_np)
    hsv_image = image.convert('HSV')
    hsv_np = np.array(hsv_image)
    unique_hues = len(np.unique(hsv_np[:, :, 0]))
    contrast = np.std(image_np)

    # 캐릭터 평가 기준별 임시 추론 (정규화된 수치 활용)
    shape_score = np.clip(contrast / 127, 0, 1) * 100
    detail_score = np.clip(brightness / 255, 0, 1) * 100
    natural_score = np.clip(unique_hues / 100, 0, 1) * 100
    color_score = np.clip((unique_hues + contrast) / 200, 0, 1) * 100

    # 설명 자동 생성
    description = "이 그림은 캐릭터를 중심으로 구성된 창작 일러스트로 보입니다. 주요 색상이 명확하게 표현되었고, 형태와 선 묘사에서 캐릭터의 감정이나 특징이 드러나는 그림입니다."

    # 피드백
    feedback = []
    if shape_score < 40:
        feedback.append("🔶 형태 표현이 다소 부족해 보입니다. 캐릭터의 외곽선이나 비율을 좀 더 정확하게 표현해보세요.")
    elif shape_score < 70:
        feedback.append("🟡 형태는 전반적으로 안정적이나 세부 조정이 필요합니다.")
    else:
        feedback.append("✅ 형태 표현이 명확하고 안정적입니다. 비례와 구성이 잘 정리되어 있어요.")

    if detail_score < 40:
        feedback.append("🔍 묘사력이 낮게 나타납니다. 눈, 옷 주름 등 세부 표현을 강화해보세요.")
    elif detail_score < 70:
        feedback.append("📝 디테일 표현이 적절하지만, 더 다양한 텍스처 표현이 있으면 좋겠습니다.")
    else:
        feedback.append("🖋️ 세부 묘사가 훌륭합니다. 작은 요소도 섬세하게 표현했어요.")

    if natural_score < 40:
        feedback.append("🧍 캐릭터 포즈나 표정이 다소 부자연스러울 수 있습니다. 동세나 감정을 더 고려해보세요.")
    elif natural_score < 70:
        feedback.append("🙂 자연스러움은 보통 수준입니다. 인체 동세나 흐름을 조금 더 살리면 좋아요.")
    else:
        feedback.append("🎯 캐릭터가 매우 자연스럽게 표현되었어요. 생동감이 느껴집니다.")

    if color_score < 40:
        feedback.append("🎨 색상 선택이 제한적입니다. 명도, 채도를 다양하게 활용해보세요.")
    elif color_score < 70:
        feedback.append("🎨 색상 구성이 무난하지만, 포인트 색을 추가하면 더 생동감이 살 수 있어요.")
    else:
        feedback.append("🌈 색상 조화와 표현이 탁월합니다. 시선이 머무는 구성이에요.")

    return description, round((shape_score + detail_score + natural_score + color_score) / 4, 1), feedback

# ===== 업로드 인터페이스 =====
uploaded_file = st.file_uploader("캐릭터 그림 사진 업로드 (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    # 1. 이미지 유형 판별
    image_type = classify_image_type(image)
    if image_type == "사진":
        st.error("⚠️ 업로드된 이미지는 사진으로 판단됩니다. 그림 파일만 평가할 수 있어요.")
    else:
        # 2. 캐릭터 그림 평가 실행
        description, score, feedback = evaluate_character_art(image)

        st.subheader("🖼️ 그림 해설")
        st.markdown(description)

        st.subheader(f"✅ 총점: {score}점 (100점 만점)")
        st.markdown("---")
        st.subheader("📝 AI 피드백")
        for f in feedback:
            st.markdown(f"- {f}")
else:
    st.info("왼쪽 사이드바에서 캐릭터 그림 이미지를 업로드해 주세요.")
