# -*- coding: utf-8 -*-
"""OpenAI API 핸들러"""
from typing import List, Optional, Dict, Union
from PIL import Image

from src.vlm.handlers.base_handler import BaseHandler
from src.utils.image_utils import encode_image_base64, resize_image


class OpenAIHandler(BaseHandler):
    """OpenAI API 기반 VLM 핸들러"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        OpenAI 핸들러 초기화
        
        Args:
            api_key: OpenAI API 키 (None이면 설정에서 가져옴)
            **kwargs: 추가 옵션
        """
        super().__init__(**kwargs)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지가 설치되지 않았습니다. pip install openai를 실행하세요.")
        
        from src.config.settings import Settings
        self.api_key = api_key or Settings.get_openai_key()
        
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=self.api_key)
        self._initialized = True
    
    def generate(
        self,
        images: Optional[List[Image.Image]],
        prompt: Union[str, Dict[str, str]],
        **kwargs
    ) -> str:
        """
        OpenAI API를 사용하여 텍스트 생성
        
        Args:
            images: PIL Image 리스트 (0개 이상)
            prompt: 프롬프트 (str 또는 Dict)
            **kwargs: 추가 옵션 (max_tokens 등)
        
        Returns:
            생성된 텍스트
        """
        self._ensure_initialized()
        
        # 이미지 검증
        validated_images = self._validate_images(images)
        
        # 프롬프트 포맷팅
        formatted_prompt = self._format_prompt(prompt)
        
        # max_tokens 옵션
        max_tokens = kwargs.get('max_tokens', 500)
        
        messages = []
        
        # 이미지가 있는 경우
        if validated_images and len(validated_images) > 0:
            content = []
            for img in validated_images:
                # 이미지 리사이즈 (API 제한 고려)
                img_resized = resize_image(img, max_size=1024)
                img_base64 = encode_image_base64(img_resized)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            
            # 텍스트 프롬프트 추가
            content.append({
                "type": "text",
                "text": formatted_prompt
            })
            
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            # 이미지가 없는 경우 텍스트만
            messages.append({
                "role": "user",
                "content": formatted_prompt
            })
        
        # GPT-4 Vision 모델 사용
        response = self.client.chat.completions.create(
            model=kwargs.get('model', 'gpt-4o'),
            messages=messages,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()


