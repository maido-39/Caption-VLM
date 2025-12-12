# -*- coding: utf-8 -*-
"""설정 관리 모듈"""
import os
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Settings:
    """애플리케이션 설정 관리"""
    
    # API 키
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # 로컬 VLM 모델 설정
    LOCAL_VLM_MODEL: str = os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen-VL-Chat")
    
    @classmethod
    def get_openai_key(cls) -> Optional[str]:
        """OpenAI API 키 반환"""
        return cls.OPENAI_API_KEY
    
    @classmethod
    def get_gemini_key(cls) -> Optional[str]:
        """Gemini API 키 반환"""
        return cls.GEMINI_API_KEY
    
    @classmethod
    def get_local_model(cls) -> str:
        """로컬 VLM 모델 이름 반환"""
        return cls.LOCAL_VLM_MODEL
    
    @classmethod
    def set_openai_key(cls, api_key: str) -> None:
        """OpenAI API 키 설정"""
        cls.OPENAI_API_KEY = api_key
    
    @classmethod
    def set_gemini_key(cls, api_key: str) -> None:
        """Gemini API 키 설정"""
        cls.GEMINI_API_KEY = api_key
    
    @classmethod
    def set_local_model(cls, model_name: str) -> None:
        """로컬 VLM 모델 이름 설정"""
        cls.LOCAL_VLM_MODEL = model_name

