# -*- coding: utf-8 -*-
"""VLM 통합 유틸리티"""
from typing import List, Optional, Callable, Dict, Any
from PIL import Image
import base64
import io

from src.config.settings import Settings
from src.utils.image_utils import process_images, encode_image_base64, resize_image


def load_openai(api_key: Optional[str] = None) -> Callable[[Optional[List[Image.Image]], str], str]:
    """
    OpenAI API 기반 VLM 로더
    
    Args:
        api_key: OpenAI API 키 (None이면 설정에서 가져옴)
    
    Returns:
        캡션 생성 함수
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai 패키지가 설치되지 않았습니다. pip install openai를 실행하세요.")
    
    key = api_key or Settings.get_openai_key()
    if not key:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    client = OpenAI(api_key=key)
    
    def generate_caption(images: Optional[List[Image.Image]], prompt: str) -> str:
        """
        OpenAI API를 사용하여 캡션 생성
        
        Args:
            images: 이미지 리스트 (0~5개, None 허용)
            prompt: 프롬프트
        
        Returns:
            생성된 캡션
        """
        messages = []
        
        # 이미지가 있는 경우
        if images and len(images) > 0:
            content = []
            for img in images:
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
                "text": prompt
            })
            
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            # 이미지가 없는 경우 텍스트만
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        # GPT-4 Vision 모델 사용
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    return generate_caption


def load_gemini(api_key: Optional[str] = None) -> Callable[[Optional[List[Image.Image]], str], str]:
    """
    Gemini API 기반 VLM 로더
    
    Args:
        api_key: Gemini API 키 (None이면 설정에서 가져옴)
    
    Returns:
        캡션 생성 함수
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai 패키지가 설치되지 않았습니다. pip install google-generativeai를 실행하세요.")
    
    key = api_key or Settings.get_gemini_key()
    if not key:
        raise ValueError("Gemini API 키가 설정되지 않았습니다.")
    
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def generate_caption(images: Optional[List[Image.Image]], prompt: str) -> str:
        """
        Gemini API를 사용하여 캡션 생성
        
        Args:
            images: 이미지 리스트 (0~5개, None 허용)
            prompt: 프롬프트
        
        Returns:
            생성된 캡션
        """
        # 이미지가 있는 경우
        if images and len(images) > 0:
            # PIL Image를 bytes로 변환
            image_parts = []
            for img in images:
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
            contents = image_parts + [prompt]
            response = model.generate_content(contents)
        else:
            # 이미지가 없는 경우 텍스트만
            response = model.generate_content(prompt)
        
        return response.text.strip()
    
    return generate_caption


def load_local(model_name: Optional[str] = None) -> Callable[[Optional[List[Image.Image]], str], str]:
    """
    로컬 VLM 로더 (Qwen 등)
    
    Args:
        model_name: 모델 이름 (None이면 설정에서 가져옴)
    
    Returns:
        캡션 생성 함수
    """
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        # qwen-vl-utils가 있으면 사용, 없으면 직접 처리
        try:
            from qwen_vl_utils import process_vision_info
            USE_QWEN_VL_UTILS = True
        except ImportError:
            USE_QWEN_VL_UTILS = False
    except ImportError:
        raise ImportError("transformers와 torch 패키지가 설치되지 않았습니다. pip install transformers torch를 실행하세요.")
    
    model_name = model_name or Settings.get_local_model()
    
    # 모델과 프로세서 로드
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to("cpu")
    
    def generate_caption(images: Optional[List[Image.Image]], prompt: str) -> str:
        """
        로컬 VLM을 사용하여 캡션 생성
        
        Args:
            images: 이미지 리스트 (0~5개, None 허용)
            prompt: 프롬프트
        
        Returns:
            생성된 캡션
        """
        # 이미지가 있는 경우
        if images and len(images) > 0:
            # Qwen-VL은 여러 이미지를 지원하므로 리스트로 전달
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in images],
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            # 이미지가 없는 경우 텍스트만
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        
        # 텍스트 프롬프트 생성
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 이미지 처리
        if images and len(images) > 0:
            if USE_QWEN_VL_UTILS:
                # qwen-vl-utils 사용
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
            else:
                # 최신 transformers API: processor에 messages 직접 전달
                inputs = processor(
                    text=[text],
                    images=images,
                    padding=True,
                    return_tensors="pt"
                )
        else:
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
        
        # 디바이스로 이동
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 생성
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 디코딩
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()
    
    return generate_caption


def generate_caption(
    vlm_type: str,
    images: Optional[List[Image.Image]],
    prompt: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    통합 캡션 생성 함수
    
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
    
    # VLM 타입에 따라 로더 선택
    if vlm_type.lower() == "openai":
        generate_func = load_openai(api_key=api_key)
    elif vlm_type.lower() == "gemini":
        generate_func = load_gemini(api_key=api_key)
    elif vlm_type.lower() == "local":
        generate_func = load_local(model_name=model_name)
    else:
        raise ValueError(f"지원하지 않는 VLM 타입: {vlm_type}. 'openai', 'gemini', 'local' 중 하나를 선택하세요.")
    
    # 캡션 생성
    return generate_func(processed_images, prompt)


# VLM 타입별 로더 딕셔너리 (편의용)
VLM_LOADERS: Dict[str, Callable] = {
    "openai": load_openai,
    "gemini": load_gemini,
    "local": load_local
}

