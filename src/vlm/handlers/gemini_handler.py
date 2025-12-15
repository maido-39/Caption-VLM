# -*- coding: utf-8 -*-
"""Gemini API VLM 핸들러"""
import io
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
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
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
        
        Returns:
            생성된 텍스트
        """
        self._ensure_initialized()
        
        # 이미지 검증
        validated_images = self._validate_images(images)
        
        # 프롬프트 포맷팅
        formatted_prompt = self._format_prompt(prompt)
        
        # 이미지가 있는 경우
        if validated_images and len(validated_images) > 0:
            # PIL Image를 bytes로 변환
            image_parts = []
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
            
            # 이미지와 텍스트 프롬프트 결합
            contents = image_parts + [formatted_prompt]
            response = self.model.generate_content(contents)
        else:
            # 이미지가 없는 경우 텍스트만
            response = self.model.generate_content(formatted_prompt)
        
        return response.text.strip()
