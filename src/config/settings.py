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
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_TEMPERATURE: Optional[float] = (
        float(os.getenv("OPENAI_TEMPERATURE")) if os.getenv("OPENAI_TEMPERATURE") else None
    )
    # 토큰 제한 설정
    # Reasoning 모델(gpt-5, o1, o3)용: max_completion_tokens
    OPENAI_MAX_COMPLETION_TOKENS: Optional[int] = (
        int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS")) if os.getenv("OPENAI_MAX_COMPLETION_TOKENS") else None
    )
    # 일반 모델(gpt-4o, gpt-4 등)용: max_tokens
    OPENAI_MAX_TOKENS: Optional[int] = (
        int(os.getenv("OPENAI_MAX_TOKENS")) if os.getenv("OPENAI_MAX_TOKENS") else None
    )
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # 로컬 VLM 모델 설정
    LOCAL_VLM_MODEL: str = os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen-VL-Chat")
    
    @classmethod
    def get_openai_key(cls) -> Optional[str]:
        """OpenAI API 키 반환"""
        return cls.OPENAI_API_KEY
    
    @classmethod
    def get_openai_model(cls) -> str:
        """OpenAI 모델 이름 반환"""
        return cls.OPENAI_MODEL
    
    @classmethod
    def get_openai_temperature(cls) -> Optional[float]:
        """OpenAI temperature 반환"""
        return cls.OPENAI_TEMPERATURE
    
    @classmethod
    def get_gemini_key(cls) -> Optional[str]:
        """Gemini API 키 반환"""
        return cls.GEMINI_API_KEY
    
    @classmethod
    def get_gemini_model(cls) -> str:
        """Gemini 모델 이름 반환"""
        return cls.GEMINI_MODEL
    
    @classmethod
    def get_local_model(cls) -> str:
        """로컬 VLM 모델 이름 반환"""
        return cls.LOCAL_VLM_MODEL
    
    @classmethod
    def set_openai_key(cls, api_key: str) -> None:
        """OpenAI API 키 설정"""
        cls.OPENAI_API_KEY = api_key
    
    @classmethod
    def set_openai_model(cls, model: str) -> None:
        """OpenAI 모델 이름 설정"""
        cls.OPENAI_MODEL = model
    
    @classmethod
    def set_openai_temperature(cls, temperature: float) -> None:
        """OpenAI temperature 설정"""
        cls.OPENAI_TEMPERATURE = temperature
    
    @classmethod
    def set_gemini_key(cls, api_key: str) -> None:
        """Gemini API 키 설정"""
        cls.GEMINI_API_KEY = api_key
    
    @classmethod
    def set_gemini_model(cls, model: str) -> None:
        """Gemini 모델 이름 설정"""
        cls.GEMINI_MODEL = model
    
    @classmethod
    def set_local_model(cls, model_name: str) -> None:
        """로컬 VLM 모델 이름 설정"""
        cls.LOCAL_VLM_MODEL = model_name
    
    @classmethod
    def get_openai_max_completion_tokens(cls) -> Optional[int]:
        """OpenAI max_completion_tokens 반환 (Reasoning 모델용)"""
        return cls.OPENAI_MAX_COMPLETION_TOKENS
    
    @classmethod
    def get_openai_max_tokens(cls) -> Optional[int]:
        """OpenAI max_tokens 반환 (일반 모델용)"""
        return cls.OPENAI_MAX_TOKENS

