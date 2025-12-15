# -*- coding: utf-8 -*-
"""
VLM 래퍼 간단한 사용 예제

이 파일은 VLMManager의 기본적인 사용법을 보여줍니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vlm import VLMManager
from PIL import Image
import os


def example_1_simple_text():
    """예제 1: 단순 텍스트만 사용"""
    print("예제 1: 단순 텍스트만 사용")
    print("-" * 50)
    
    manager = VLMManager()
    
    result = manager.call_vlm(
        vlm_name="openai",
        images=None,
        prompt="고양이에 대해 한 문장으로 설명해주세요.",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"결과: {result}\n")


def example_2_with_image():
    """예제 2: 이미지와 함께 사용"""
    print("예제 2: 이미지와 함께 사용")
    print("-" * 50)
    
    manager = VLMManager()
    
    # 이미지 파일 경로 (실제 경로로 변경하세요)
    image_path = "test_image.jpg"
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        
        result = manager.call_vlm(
            vlm_name="openai",
            images=[image],
            prompt="이 이미지를 자세히 설명해주세요.",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print(f"결과: {result}\n")
    else:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}\n")


def example_3_structured_prompt():
    """예제 3: 구조화된 프롬프트 사용"""
    print("예제 3: 구조화된 프롬프트 사용")
    print("-" * 50)
    
    manager = VLMManager()
    
    prompt_dict = {
        "system_prompt": "당신은 전문 이미지 분석가입니다.",
        "user_prompt": "이 이미지의 주요 특징을 나열해주세요."
    }
    
    result = manager.call_vlm(
        vlm_name="openai",
        images=None,
        prompt=prompt_dict,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"결과: {result}\n")


def example_4_different_vlms():
    """예제 4: 다른 VLM 타입 사용"""
    print("예제 4: 다른 VLM 타입 사용")
    print("-" * 50)
    
    manager = VLMManager()
    
    prompt = "파이썬에 대해 한 문장으로 설명해주세요."
    
    # OpenAI
    try:
        result = manager.call_vlm(
            vlm_name="openai",
            images=None,
            prompt=prompt,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print(f"OpenAI: {result}")
    except Exception as e:
        print(f"OpenAI 오류: {e}")
    
    # Gemini
    try:
        result = manager.call_vlm(
            vlm_name="gemini",
            images=None,
            prompt=prompt,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        print(f"Gemini: {result}")
    except Exception as e:
        print(f"Gemini 오류: {e}")
    
    # Local (Qwen)
    try:
        result = manager.call_vlm(
            vlm_name="local",
            images=None,
            prompt=prompt,
            model_name=os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen-VL-Chat")
        )
        print(f"Local: {result}")
    except Exception as e:
        print(f"Local 오류: {e}")
    
    print()


def example_5_with_options():
    """예제 5: 추가 옵션 사용"""
    print("예제 5: 추가 옵션 사용")
    print("-" * 50)
    
    manager = VLMManager()
    
    # OpenAI with max_tokens
    result = manager.call_vlm(
        vlm_name="openai",
        images=None,
        prompt="고양이에 대해 설명해주세요.",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=50  # 짧은 응답 요청
    )
    
    print(f"결과 (max_tokens=50): {result}\n")


def example_6_backward_compatibility():
    """예제 6: 기존 함수 사용 (하위 호환성)"""
    print("예제 6: 기존 함수 사용 (하위 호환성)")
    print("-" * 50)
    
    from src.vlm_utils import generate_caption
    
    result = generate_caption(
        vlm_type="openai",
        images=None,
        prompt="하위 호환성 테스트입니다.",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"결과: {result}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("VLM 래퍼 간단한 사용 예제")
    print("=" * 60)
    print()
    
    # 각 예제 실행
    example_1_simple_text()
    example_2_with_image()
    example_3_structured_prompt()
    example_4_different_vlms()
    example_5_with_options()
    example_6_backward_compatibility()
    
    print("=" * 60)
    print("모든 예제 완료")
    print("=" * 60)

