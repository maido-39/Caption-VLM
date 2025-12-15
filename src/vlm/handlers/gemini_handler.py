# -*- coding: utf-8 -*-
"""Gemini API VLM 핸들러"""
import io
import re
import time
from typing import List, Optional, Dict, Union
from PIL import Image

from src.vlm.handlers.base_handler import BaseHandler
from src.utils.image_utils import resize_image


class GeminiHandler(BaseHandler):
    """Gemini API 기반 VLM 핸들러"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Gemini 핸들러 초기화
        
        Args:
            api_key: Gemini API 키 (None이면 설정에서 가져옴)
            **kwargs: 추가 옵션
        """
        super().__init__(**kwargs)
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai 패키지가 설치되지 않았습니다. pip install google-generativeai를 실행하세요.")
        
        from src.config.settings import Settings
        
        self.api_key = api_key or Settings.get_gemini_key()
        if not self.api_key:
            raise ValueError("Gemini API 키가 설정되지 않았습니다.")
        
        genai.configure(api_key=self.api_key)
        # 모델 이름을 kwargs에서 가져오거나 설정에서 가져옴
        model_name = kwargs.get('model_name') or Settings.get_gemini_model()
        self.model = genai.GenerativeModel(model_name)
        self._initialized = True
    
    def generate(
        self,
        images: Optional[List[Image.Image]],
        prompt: Union[str, Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Gemini API를 사용하여 텍스트 생성
        
        Args:
            images: PIL Image 리스트 (0개 이상)
            prompt: 프롬프트 (str 또는 Dict)
            **kwargs: 추가 옵션
                - max_retries: 최대 재시도 횟수 (기본: 3)
                - retry_delay: 재시도 대기 시간 (초, 기본: None, 에러 메시지에서 자동 파싱)
        
        Returns:
            생성된 텍스트
        """
        self._ensure_initialized()
        
        # 재시도 설정
        max_retries = kwargs.get('max_retries', 3)
        base_retry_delay = kwargs.get('retry_delay', None)
        
        # 이미지 검증
        validated_images = self._validate_images(images)
        
        # 프롬프트 포맷팅
        formatted_prompt = self._format_prompt(prompt)
        
        # 이미지 준비
        image_parts = []
        if validated_images and len(validated_images) > 0:
            for img in validated_images:
                img_resized = resize_image(img, max_size=1024)
                buffered = io.BytesIO()
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                img_resized.save(buffered, format="JPEG")
                image_parts.append({
                    "mime_type": "image/jpeg",
                    "data": buffered.getvalue()
                })
        
        # 재시도 로직
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # 이미지가 있는 경우
                if image_parts:
                    contents = image_parts + [formatted_prompt]
                    response = self.model.generate_content(contents)
                else:
                    # 이미지가 없는 경우 텍스트만
                    response = self.model.generate_content(formatted_prompt)
                
                return response.text.strip()
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 429 에러 (할당량 초과) 처리
                if '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower():
                    # 재시도 대기 시간 파싱
                    retry_after = self._parse_retry_after(error_str, base_retry_delay)
                    
                    if attempt < max_retries:
                        # 재시도 대기 시간에 약간의 랜덤 지터 추가 (동시 요청 방지)
                        import random
                        jitter = random.uniform(0.1, 0.5)
                        wait_time = retry_after + jitter
                        
                        print(f"[WARNING] 할당량 초과 에러 발생. {wait_time:.2f}초 후 재시도 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # 최대 재시도 횟수 초과
                        raise Exception(
                            f"할당량 초과 에러가 {max_retries}번 재시도 후에도 해결되지 않았습니다. "
                            f"에러 메시지: {error_str}"
                        ) from e
                else:
                    # 429가 아닌 다른 에러는 즉시 발생
                    raise
        
        # 여기 도달하면 안 되지만 안전장치
        if last_error:
            raise last_error
        raise RuntimeError("예상치 못한 오류가 발생했습니다.")
    
    def _parse_retry_after(self, error_str: str, default_delay: Optional[float] = None) -> float:
        """
        에러 메시지에서 재시도 대기 시간 파싱
        
        Args:
            error_str: 에러 메시지 문자열
            default_delay: 기본 대기 시간 (초)
        
        Returns:
            재시도 대기 시간 (초)
        """
        # "Please retry in X.XXXXXXs" 패턴 찾기
        retry_pattern = r'retry in ([\d.]+)s'
        match = re.search(retry_pattern, error_str, re.IGNORECASE)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 기본값 사용
        if default_delay is not None:
            return default_delay
        
        # 기본값도 없으면 5초
        return 5.0
