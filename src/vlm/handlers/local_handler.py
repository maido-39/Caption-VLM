# -*- coding: utf-8 -*-
"""로컬 VLM 핸들러 (Qwen 등)"""
from typing import List, Optional, Dict, Union
from PIL import Image

from src.vlm.handlers.base_handler import BaseHandler


class LocalHandler(BaseHandler):
    """로컬 VLM 기반 핸들러 (Qwen-VL 등)"""
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """
        로컬 VLM 핸들러 초기화
        
        Args:
            model_name: 모델 이름 (None이면 설정에서 가져옴)
            **kwargs: 추가 옵션
        """
        super().__init__(**kwargs)
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
        except ImportError:
            raise ImportError("transformers와 torch 패키지가 설치되지 않았습니다. pip install transformers torch를 실행하세요.")
        
        from src.config.settings import Settings
        
        self.model_name = model_name or Settings.get_local_model()
        self.torch = torch
        
        # qwen-vl-utils가 있으면 사용, 없으면 직접 처리
        try:
            from qwen_vl_utils import process_vision_info
            self.use_qwen_vl_utils = True
            self.process_vision_info = process_vision_info
        except ImportError:
            self.use_qwen_vl_utils = False
            self.process_vision_info = None
        
        # 모델과 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        
        self._initialized = True
    
    def generate(
        self,
        images: Optional[List[Image.Image]],
        prompt: Union[str, Dict[str, str]],
        **kwargs
    ) -> str:
        """
        로컬 VLM을 사용하여 텍스트 생성
        
        Args:
            images: PIL Image 리스트 (0개 이상)
            prompt: 프롬프트 (str 또는 Dict)
            **kwargs: 추가 옵션 (max_new_tokens 등)
        
        Returns:
            생성된 텍스트
        """
        self._ensure_initialized()
        
        # 이미지 검증
        validated_images = self._validate_images(images)
        
        # 프롬프트 포맷팅
        formatted_prompt = self._format_prompt(prompt)
        
        # max_new_tokens 옵션
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        
        # 메시지 구성
        if validated_images and len(validated_images) > 0:
            # Qwen-VL은 여러 이미지를 지원하므로 리스트로 전달
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in validated_images],
                        {"type": "text", "text": formatted_prompt}
                    ]
                }
            ]
        else:
            # 이미지가 없는 경우 텍스트만
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt}
                    ]
                }
            ]
        
        # Qwen2-VL API 처리
        # apply_chat_template을 사용하여 텍스트 생성 후 processor에 전달
        try:
            # apply_chat_template로 텍스트 생성
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # processor에 text와 images 전달
            if validated_images and len(validated_images) > 0:
                if self.use_qwen_vl_utils:
                    # qwen-vl-utils 사용 (비디오 처리 지원)
                    try:
                        image_inputs, video_inputs = self.process_vision_info(messages)
                        inputs = self.processor(
                            text=text,
                            images=image_inputs if image_inputs else validated_images,
                            videos=video_inputs if video_inputs else None,
                            padding=True,
                            return_tensors="pt"
                        )
                    except Exception:
                        # qwen-vl-utils 실패 시 직접 이미지 전달
                        inputs = self.processor(
                            text=text,
                            images=validated_images,
                            padding=True,
                            return_tensors="pt"
                        )
                else:
                    # 직접 이미지 전달
                    inputs = self.processor(
                        text=text,
                        images=validated_images,
                        padding=True,
                        return_tensors="pt"
                    )
            else:
                # 이미지가 없는 경우 텍스트만
                inputs = self.processor(
                    text=text,
                    padding=True,
                    return_tensors="pt"
                )
        except Exception as e:
            raise RuntimeError(f"Qwen2-VL API 호출 실패: {str(e)}")
        
        # 디바이스로 이동
        inputs = {
            k: v.to(self.model.device) if isinstance(v, self.torch.Tensor) else v 
            for k, v in inputs.items()
        }
        
        # 생성
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # input_ids는 dict에서 가져오기
        if 'input_ids' not in inputs:
            raise ValueError("processor 출력에 'input_ids' 키가 없습니다.")
        
        input_ids = inputs['input_ids']
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        # 디코딩
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()
