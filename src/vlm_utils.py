# -*- coding: utf-8 -*-
"""VLM 통합 유틸리티 (하위 호환성을 위한 래퍼)"""
from typing import List, Optional, Callable, Dict, Any
from PIL import Image

from src.vlm import VLMManager
from src.utils.image_utils import process_images


def load_openai(api_key: Optional[str] = None) -> Callable[[Optional[List[Image.Image]], str], str]:
    """
    OpenAI API 기반 VLM 로더 (하위 호환성 유지)
    
    Args:
        api_key: OpenAI API 키 (None이면 설정에서 가져옴)
    
    Returns:
        캡션 생성 함수
    """
    manager = VLMManager()
    
    def generate_caption(images: Optional[List[Image.Image]], prompt: str) -> str:
        """
        OpenAI API를 사용하여 캡션 생성
        
        Args:
            images: 이미지 리스트 (0~5개, None 허용)
            prompt: 프롬프트
        
        Returns:
            생성된 캡션
        """
        return manager.call_vlm(
            vlm_name="openai",
            images=images,
            prompt=prompt,
            api_key=api_key
        )
    
    return generate_caption


def load_gemini(api_key: Optional[str] = None) -> Callable[[Optional[List[Image.Image]], str], str]:
    """
    Gemini API 기반 VLM 로더 (하위 호환성 유지)
    
    Args:
        api_key: Gemini API 키 (None이면 설정에서 가져옴)
    
    Returns:
        캡션 생성 함수
    """
    manager = VLMManager()
    
    def generate_caption(images: Optional[List[Image.Image]], prompt: str) -> str:
        """
        Gemini API를 사용하여 캡션 생성
        
        Args:
            images: 이미지 리스트 (0~5개, None 허용)
            prompt: 프롬프트
        
        Returns:
            생성된 캡션
        """
        return manager.call_vlm(
            vlm_name="gemini",
            images=images,
            prompt=prompt,
            api_key=api_key
        )
    
    return generate_caption


def load_local(model_name: Optional[str] = None) -> Callable[[Optional[List[Image.Image]], str], str]:
    """
    로컬 VLM 로더 (Qwen 등) (하위 호환성 유지)
    
    Args:
        model_name: 모델 이름 (None이면 설정에서 가져옴)
    
    Returns:
        캡션 생성 함수
    """
    manager = VLMManager()
    
    def generate_caption(images: Optional[List[Image.Image]], prompt: str) -> str:
        """
        로컬 VLM을 사용하여 캡션 생성
        
        Args:
            images: 이미지 리스트 (0~5개, None 허용)
            prompt: 프롬프트
        
        Returns:
            생성된 캡션
        """
        return manager.call_vlm(
            vlm_name="local",
            images=images,
            prompt=prompt,
            model_name=model_name
        )
    
    return generate_caption


def generate_caption(
    vlm_type: str,
    images: Optional[List[Image.Image]],
    prompt: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    통합 캡션 생성 함수 (하위 호환성 유지)
    
    Args:
        vlm_type: VLM 타입 ("openai", "gemini", "local")
        images: 이미지 리스트 (0~5개, None 허용)
        prompt: 프롬프트
        api_key: API 키 (선택적)
        model_name: 로컬 모델 이름 (선택적)
    
    Returns:
        생성된 캡션
    """
    # 이미지 처리
    processed_images = process_images(images)
    
    # VLMManager 사용
    manager = VLMManager()
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if model_name:
        kwargs["model_name"] = model_name
    
    return manager.call_vlm(
        vlm_name=vlm_type,
        images=processed_images,
        prompt=prompt,
        **kwargs
    )


# VLM 타입별 로더 딕셔너리 (편의용)
VLM_LOADERS: Dict[str, Callable] = {
    "openai": load_openai,
    "gemini": load_gemini,
    "local": load_local
}
