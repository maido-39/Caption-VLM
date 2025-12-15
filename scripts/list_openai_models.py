#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API로 호출 가능한 모델 목록을 확인하는 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings


def list_openai_models():
    """OpenAI API로 사용 가능한 모델 목록 출력"""
    try:
        from openai import OpenAI
    except ImportError:
        print("오류: openai 패키지가 설치되지 않았습니다.")
        print("설치: pip install openai")
        return
    
    api_key = Settings.get_openai_key()
    if not api_key:
        print("오류: OpenAI API 키가 설정되지 않았습니다.")
        print(".env 파일에 OPENAI_API_KEY를 설정하거나 환경 변수로 설정하세요.")
        return
    
    client = OpenAI(api_key=api_key)
    
    try:
        models = client.models.list()
        
        print("=" * 80)
        print("OpenAI API 사용 가능한 모델 목록")
        print("=" * 80)
        print()
        
        # Vision 모델 (이미지 처리 가능)
        vision_models = []
        # Chat 모델
        chat_models = []
        # 기타 모델
        other_models = []
        
        for model in models.data:
            model_id = model.id
            # Vision 모델 필터링 (gpt-4o, gpt-4-turbo, gpt-4-vision 등)
            if "vision" in model_id.lower() or "gpt-4o" in model_id or "gpt-4-turbo" in model_id:
                vision_models.append(model_id)
            elif "gpt" in model_id.lower() or "chat" in model_id.lower():
                chat_models.append(model_id)
            else:
                other_models.append(model_id)
        
        if vision_models:
            print("📸 Vision 모델 (이미지 처리 가능):")
            for model in sorted(vision_models):
                print(f"  - {model}")
            print()
        
        if chat_models:
            print("💬 Chat 모델:")
            for model in sorted(chat_models):
                print(f"  - {model}")
            print()
        
        if other_models:
            print("🔧 기타 모델:")
            for model in sorted(other_models):
                print(f"  - {model}")
            print()
        
        print(f"총 {len(models.data)}개 모델")
        print()
        print("=" * 80)
        print("참고: Vision 모델만 이미지 캡션 생성에 사용 가능합니다.")
        print("주요 Vision 모델:")
        print("  - gpt-4o (권장)")
        print("  - gpt-4o-mini")
        print("  - gpt-4-turbo")
        print("  - gpt-4-vision-preview")
        print("=" * 80)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\nAPI 키가 유효한지 확인하세요.")


if __name__ == "__main__":
    list_openai_models()

