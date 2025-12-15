# -*- coding: utf-8 -*-
"""VLM 핸들러 기본 클래스"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union
from PIL import Image


class BaseHandler(ABC):
    """VLM 핸들러 기본 추상 클래스"""
    
    def __init__(self, **kwargs):
        """
        핸들러 초기화
        
        Args:
            **kwargs: 핸들러별 설정 옵션
        """
        self._initialized = False
    
    @abstractmethod
    def generate(
        self,
        images: Optional[List[Image.Image]],
        prompt: Union[str, Dict[str, str]],
        **kwargs
    ) -> str:
        """
        VLM을 사용하여 텍스트 생성 (추상 메서드)
        
        Args:
            images: PIL Image 리스트 (0개 이상, 선택)
            prompt: 프롬프트
                - str: 단순 텍스트 프롬프트
                - Dict: 구조화된 프롬프트
                    - system_prompt: 시스템 프롬프트 (선택)
                    - context: 컨텍스트 (선택)
                    - user_prompt: 사용자 프롬프트 (필수 또는 prompt 키 사용)
                    - prompt: 사용자 프롬프트 (user_prompt가 없을 때 사용)
            **kwargs: 추가 옵션 (모델별 파라미터 등)
        
        Returns:
            생성된 텍스트
        """
        pass
    
    def _validate_images(self, images: Optional[List[Image.Image]]) -> Optional[List[Image.Image]]:
        """
        이미지 검증
        
        Args:
            images: 이미지 리스트
        
        Returns:
            검증된 이미지 리스트 또는 None
        """
        if images is None:
            return None
        
        if not isinstance(images, list):
            raise ValueError(f"images는 리스트여야 합니다. 받은 타입: {type(images)}")
        
        validated = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise ValueError(f"이미지는 PIL.Image.Image 타입이어야 합니다. 받은 타입: {type(img)}")
            validated.append(img)
        
        return validated if validated else None
    
    def _format_prompt(self, prompt: Union[str, Dict[str, str]]) -> str:
        """
        프롬프트를 문자열로 포맷팅
        
        Args:
            prompt: 프롬프트
                - str: 단순 텍스트 프롬프트
                - Dict: 구조화된 프롬프트
        
        Returns:
            포맷팅된 프롬프트 문자열
        """
        # 문자열인 경우 그대로 반환
        if isinstance(prompt, str):
            return prompt
        
        # 딕셔너리인 경우 처리
        if not isinstance(prompt, dict):
            raise ValueError(f"prompt는 str 또는 dict여야 합니다. 받은 타입: {type(prompt)}")
        
        parts = []
        
        # 시스템 프롬프트
        if "system_prompt" in prompt and prompt["system_prompt"]:
            parts.append(str(prompt["system_prompt"]).strip())
        
        # 컨텍스트
        if "context" in prompt and prompt["context"]:
            parts.append(str(prompt["context"]).strip())
        
        # 사용자 프롬프트
        user_prompt = None
        if "user_prompt" in prompt and prompt["user_prompt"]:
            user_prompt = str(prompt["user_prompt"]).strip()
        elif "prompt" in prompt and prompt["prompt"]:
            user_prompt = str(prompt["prompt"]).strip()
        
        if user_prompt:
            parts.append(user_prompt)
        elif not parts:
            # 프롬프트가 없으면 기본값
            parts.append("이미지에 대해 설명해주세요.")
        
        return "\n\n".join(parts)
    
    def _ensure_initialized(self):
        """핸들러가 초기화되었는지 확인"""
        if not self._initialized:
            raise RuntimeError("핸들러가 초기화되지 않았습니다. 먼저 초기화를 수행하세요.")
