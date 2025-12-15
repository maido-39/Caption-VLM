# -*- coding: utf-8 -*-
"""VLM Manager - 범용 VLM 래퍼 및 핸들러"""
from typing import List, Optional, Dict, Union
from PIL import Image

from src.vlm.handlers.openai_handler import OpenAIHandler
from src.vlm.handlers.gemini_handler import GeminiHandler
from src.vlm.handlers.local_handler import LocalHandler


class VLMManager:
    """VLM 통합 관리자 (싱글톤 패턴)"""
    
    _instance = None
    _handlers = {}
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super(VLMManager, cls).__new__(cls)
        return cls._instance
    
    def call_vlm(
        self,
        vlm_name: str,
        images: Optional[List[Image.Image]],
        prompt: Union[str, Dict[str, str]],
        **kwargs
    ) -> str:
        """
        VLM 호출 (범용 메서드)
        
        Args:
            vlm_name: VLM 이름 ("openai", "gemini", "local")
            images: PIL Image 리스트 (0개 이상)
            prompt: 프롬프트
                - str: 단순 텍스트 프롬프트
                - Dict: 구조화된 프롬프트
                    - system_prompt: 시스템 프롬프트 (선택)
                    - context: 컨텍스트 (선택)
                    - user_prompt: 사용자 프롬프트 (필수 또는 prompt 키 사용)
                    - prompt: 사용자 프롬프트 (user_prompt가 없을 때 사용)
            **kwargs: 추가 옵션
                - api_key: API 키 (openai, gemini용)
                - model_name: 모델 이름 (local용)
                - 기타 모델별 파라미터 (max_tokens, max_new_tokens 등)
        
        Returns:
            생성된 텍스트
        
        Examples:
            >>> manager = VLMManager()
            >>> result = manager.call_vlm(
            ...     vlm_name="openai",
            ...     images=[img1, img2],
            ...     prompt="이 이미지들을 설명해주세요.",
            ...     api_key="sk-...",
            ...     max_tokens=500
            ... )
            
            >>> result = manager.call_vlm(
            ...     vlm_name="local",
            ...     images=[img1],
            ...     prompt={
            ...         "system_prompt": "You are a helpful assistant",
            ...         "user_prompt": "Describe this image"
            ...     },
            ...     model_name="Qwen/Qwen-VL-Chat",
            ...     max_new_tokens=512
            ... )
        """
        vlm_name = vlm_name.lower()
        
        # 핸들러 가져오기 또는 생성 (초기화 옵션만 전달)
        handler_kwargs = {}
        if vlm_name == "openai":
            handler_kwargs['api_key'] = kwargs.get('api_key')
        elif vlm_name == "gemini":
            handler_kwargs['api_key'] = kwargs.get('api_key')
            handler_kwargs['model_name'] = kwargs.get('model_name')  # Gemini 모델 이름 지원
        elif vlm_name == "local":
            handler_kwargs['model_name'] = kwargs.get('model_name')
        
        handler = self._get_handler(vlm_name, **handler_kwargs)
        
        # VLM 호출 (초기화 옵션 제외한 나머지 kwargs만 전달)
        # Gemini의 경우 model_name은 초기화 옵션이므로 generate_kwargs에서 제외
        generate_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['api_key', 'model_name']}
        
        return handler.generate(images, prompt, **generate_kwargs)
    
    def _get_handler(self, vlm_name: str, **kwargs) -> Union[OpenAIHandler, GeminiHandler, LocalHandler]:
        """
        핸들러 가져오기 (캐시 사용)
        
        Args:
            vlm_name: VLM 이름
            **kwargs: 핸들러 초기화 옵션
        
        Returns:
            핸들러 인스턴스
        """
        vlm_name = vlm_name.lower()
        
        # 캐시 키 생성 (API 키나 모델 이름이 변경되면 새 인스턴스 생성)
        cache_key = self._get_cache_key(vlm_name, **kwargs)
        
        # 캐시에 있으면 반환
        if cache_key in self._handlers:
            return self._handlers[cache_key]
        
        # 새 핸들러 생성
        if vlm_name == "openai":
            handler = OpenAIHandler(api_key=kwargs.get('api_key'))
        elif vlm_name == "gemini":
            handler = GeminiHandler(api_key=kwargs.get('api_key'), model_name=kwargs.get('model_name'))
        elif vlm_name == "local":
            handler = LocalHandler(model_name=kwargs.get('model_name'))
        else:
            raise ValueError(
                f"지원하지 않는 VLM 이름: {vlm_name}. "
                f"'openai', 'gemini', 'local' 중 하나를 선택하세요."
            )
        
        # 캐시에 저장
        self._handlers[cache_key] = handler
        
        return handler
    
    def _get_cache_key(self, vlm_name: str, **kwargs) -> str:
        """
        캐시 키 생성
        
        Args:
            vlm_name: VLM 이름
            **kwargs: 핸들러 옵션
        
        Returns:
            캐시 키 문자열
        """
        if vlm_name == "openai":
            return f"openai_{kwargs.get('api_key', 'default')}"
        elif vlm_name == "gemini":
            model_name = kwargs.get('model_name', 'default')
            return f"gemini_{kwargs.get('api_key', 'default')}_{model_name}"
        elif vlm_name == "local":
            return f"local_{kwargs.get('model_name', 'default')}"
        else:
            return f"{vlm_name}_default"
    
    def clear_cache(self):
        """핸들러 캐시 초기화"""
        self._handlers.clear()
