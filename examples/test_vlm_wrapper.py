# -*- coding: utf-8 -*-
"""
VLM 래퍼 테스트 예제

이 파일은 VLMManager의 다양한 사용법을 보여줍니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vlm import VLMManager
from PIL import Image
import os


def test_basic_usage():
    """기본 사용법 테스트"""
    print("\n=== 기본 사용법 테스트 ===")
    
    
    # 단순 텍스트 프롬프트 (이미지 없음)
    try:
        result = manager.call_vlm(
            vlm_name="openai",
            images=None,
            prompt="안녕하세요. 간단히 자기소개해주세요."
        )
        print(f"OpenAI (텍스트만): {result[:100]}...")
    except Exception as e:
        print(f"OpenAI 오류: {e}")


def test_with_images():
    """이미지가 있는 경우 테스트"""
    print("\n=== 이미지가 있는 경우 테스트 ===")
    
    manager = VLMManager()
    
    # 테스트 이미지 경로 (실제 이미지 파일 경로로 변경 필요)
    test_image_path = "examples/kuroneko.jpg"  # 실제 이미지 경로로 변경하세요
    
    if os.path.exists(test_image_path):
        try:
            image = Image.open(test_image_path)
            
            result = manager.call_vlm(
                vlm_name="openai",
                images=[image],
                prompt="이 이미지를 자세히 설명해주세요."
            )
            print(f"OpenAI (이미지 포함): {result[:200]}...")
        except Exception as e:
            print(f"OpenAI 이미지 처리 오류: {e}")
    else:
        print(f"테스트 이미지를 찾을 수 없습니다: {test_image_path}")


def test_structured_prompt():
    """구조화된 프롬프트 테스트"""
    print("\n=== 구조화된 프롬프트 테스트 ===")
    
    manager = VLMManager()
    
    prompt_dict = {
        "system_prompt": "당신은 전문 이미지 분석가입니다. 이미지를 객관적이고 상세하게 분석합니다.",
        "context": "이전 분석에서 이 이미지는 자연 풍경 사진으로 확인되었습니다.",
        "user_prompt": "이 이미지의 구체적인 세부사항을 설명해주세요."
    }
    
    try:
        result = manager.call_vlm(
            vlm_name="openai",
            images=None,
            prompt=prompt_dict
        )
        print(f"구조화된 프롬프트 결과: {result[:200]}...")
    except Exception as e:
        print(f"구조화된 프롬프트 오류: {e}")


def test_multiple_vlms():
    """여러 VLM 타입 테스트"""
    print("\n=== 여러 VLM 타입 테스트 ===")
    
    manager = VLMManager()
    
    test_prompt = "고양이에 대해 한 문장으로 설명해주세요."
    
    # OpenAI 테스트
    try:
        result = manager.call_vlm(
            vlm_name="openai",
            images=None,
            prompt=test_prompt,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print(f"OpenAI: {result}")
    except Exception as e:
        print(f"OpenAI 오류: {e}")
    
    # Gemini 테스트
    try:
        result = manager.call_vlm(
            vlm_name="gemini",
            images=None,
            prompt=test_prompt,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        print(f"Gemini: {result}")
    except Exception as e:
        print(f"Gemini 오류: {e}")
    
    # Local (Qwen) 테스트
    try:
        result = manager.call_vlm(
            vlm_name="local",
            images=None,
            prompt=test_prompt,
            model_name=os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen-VL-Chat")
        )
        print(f"Local (Qwen): {result}")
    except Exception as e:
        print(f"Local 오류: {e}")


def test_multiple_images():
    """여러 이미지 테스트"""
    print("\n=== 여러 이미지 테스트 ===")
    
    manager = VLMManager()
    
    # 여러 이미지 경로 (실제 경로로 변경 필요)
    image_paths = ["examples/kuroneko.jpg", "image2.jpg", "image3.jpg"]
    images = []
    
    for path in image_paths:
        if os.path.exists(path):
            images.append(Image.open(path))
    
    if images:
        try:
            result = manager.call_vlm(
                vlm_name="openai",
                images=images,
                prompt="이 이미지들을 비교하여 공통점과 차이점을 설명해주세요."
            )
            print(f"여러 이미지 결과: {result[:300]}...")
        except Exception as e:
            print(f"여러 이미지 처리 오류: {e}")
    else:
        print("테스트 이미지를 찾을 수 없습니다.")


def test_with_options():
    """추가 옵션 사용 테스트"""
    print("\n=== 추가 옵션 사용 테스트 ===")
    
    manager = VLMManager()
    
    # OpenAI with max_tokens
    try:
        result = manager.call_vlm(
            vlm_name="openai",
            images=None,
            prompt="고양이에 대해 설명해주세요.",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=100  # 짧은 응답 요청
        )
        print(f"OpenAI (max_tokens=100): {result}")
    except Exception as e:
        print(f"OpenAI 옵션 테스트 오류: {e}")
    
    # Local with max_new_tokens
    try:
        result = manager.call_vlm(
            vlm_name="local",
            images=None,
            prompt="고양이에 대해 설명해주세요.",
            model_name=os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen-VL-Chat"),
            max_new_tokens=100  # 짧은 응답 요청
        )
        print(f"Local (max_new_tokens=100): {result}")
    except Exception as e:
        print(f"Local 옵션 테스트 오류: {e}")


def test_error_handling():
    """에러 핸들링 테스트"""
    print("\n=== 에러 핸들링 테스트 ===")
    
    manager = VLMManager()
    
    # 잘못된 VLM 이름
    try:
        result = manager.call_vlm(
            vlm_name="invalid_vlm",
            images=None,
            prompt="테스트"
        )
        print(f"결과: {result}")
    except ValueError as e:
        print(f"예상된 오류 (잘못된 VLM 이름): {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
    
    # API 키 없음 (환경 변수에 설정되어 있지 않은 경우)
    try:
        result = manager.call_vlm(
            vlm_name="openai",
            images=None,
            prompt="테스트",
            api_key=None  # API 키가 환경 변수에도 없으면 오류 발생
        )
        print(f"결과: {result}")
    except ValueError as e:
        print(f"예상된 오류 (API 키 없음): {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")


def test_singleton_pattern():
    """싱글톤 패턴 테스트"""
    print("\n=== 싱글톤 패턴 테스트 ===")
    
    manager1 = VLMManager()
    manager2 = VLMManager()
    
    # 같은 인스턴스인지 확인
    if manager1 is manager2:
        print("✓ VLMManager는 싱글톤 패턴으로 작동합니다.")
    else:
        print("✗ VLMManager 싱글톤 패턴 오류")
    
    # 캐시 초기화 테스트
    manager1.clear_cache()
    print("✓ 캐시 초기화 완료")


def test_backward_compatibility():
    """하위 호환성 테스트 (기존 vlm_utils.py 함수 사용)"""
    print("\n=== 하위 호환성 테스트 ===")
    
    from src.vlm_utils import generate_caption, load_openai, load_gemini, load_local
    
    # generate_caption 함수 사용
    try:
        result = generate_caption(
            vlm_type="openai",
            images=None,
            prompt="하위 호환성 테스트입니다.",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print(f"generate_caption 함수: {result[:100]}...")
    except Exception as e:
        print(f"generate_caption 오류: {e}")
    
    # load_openai 함수 사용
    try:
        openai_func = load_openai(api_key=os.getenv("OPENAI_API_KEY"))
        result = openai_func(None, "load_openai 함수 테스트입니다.")
        print(f"load_openai 함수: {result[:100]}...")
    except Exception as e:
        print(f"load_openai 오류: {e}")


def main():
    """모든 테스트 실행"""
    print("=" * 60)
    print("VLM 래퍼 테스트 시작")
    print("=" * 60)
    
    # 기본 사용법
    test_basic_usage()
    
    # 이미지가 있는 경우
    test_with_images()
    
    # 구조화된 프롬프트
    test_structured_prompt()
    
    # 여러 VLM 타입
    test_multiple_vlms()
    
    # 여러 이미지
    test_multiple_images()
    
    # 추가 옵션
    test_with_options()
    
    # 에러 핸들링
    test_error_handling()
    
    # 싱글톤 패턴
    test_singleton_pattern()
    
    # 하위 호환성
    test_backward_compatibility()
    
    print("\n" + "=" * 60)
    print("모든 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()

