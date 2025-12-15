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
        
        # 모델 설정: kwargs > Settings > 기본값 (먼저 확인 필요)
        from src.config.settings import Settings
        model_name = kwargs.get('model') or Settings.get_openai_model() or "gpt-4o"
        
        # Reasoning 모델 (gpt-5, o1, o3 시리즈)은 더 큰 토큰 제한 필요
        # Reasoning 토큰 + 실제 응답 토큰을 모두 포함해야 함
        is_reasoning_model = (
            model_name.startswith("gpt-5") or
            model_name.startswith("o1") or
            model_name.startswith("o3")
        )
        
        # max_tokens/max_completion_tokens 옵션 처리
        # 최신 모델은 max_completion_tokens를 사용, 구버전은 max_tokens 사용
        max_tokens = kwargs.get('max_tokens')
        max_completion_tokens = kwargs.get('max_completion_tokens')
        
        # 둘 다 없으면 기본값 설정
        # Settings에서 환경 변수 값 확인, 없으면 하드코딩된 기본값 사용
        if max_tokens is None and max_completion_tokens is None:
            if is_reasoning_model:
                # Reasoning 모델: Settings > 기본값(1000)
                max_completion_tokens = Settings.get_openai_max_completion_tokens() or 1000
            else:
                # 일반 모델: Settings > 기본값(500)
                max_tokens = Settings.get_openai_max_tokens() or 500
        
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
        
        # temperature 파라미터: kwargs > Settings > None
        temperature = kwargs.get("temperature")
        if temperature is None:
            temperature = Settings.get_openai_temperature()
        
        # OpenAI 공식 문서 기준 모델별 파라미터 구성
        # 참고: https://platform.openai.com/docs/guides/latest-model
        # 참고: https://platform.openai.com/docs/api-reference/chat/create
        
        # 모델별 파라미터 설정 (switch 문 스타일)
        use_max_completion_tokens = False
        supports_temperature = True
        
        # 모델 이름 정규화 (버전 정보 제거)
        model_base = model_name.split("-")[0] + "-" + model_name.split("-")[1] if "-" in model_name else model_name
        
        # 모델별 파라미터 설정
        if model_name.startswith("gpt-4o") or model_name.startswith("gpt-4o-"):
            # GPT-4o 시리즈: max_completion_tokens 사용, temperature 지원
            # 예: gpt-4o, gpt-4o-2024-08-06, gpt-4o-mini
            use_max_completion_tokens = True
            supports_temperature = True
        elif model_name.startswith("gpt-5") or model_name.startswith("gpt-5."):
            # GPT-5 시리즈: max_completion_tokens 사용, temperature 미지원 (기본값 1만)
            # 예: gpt-5, gpt-5.1, gpt-5.2, gpt-5-mini
            use_max_completion_tokens = True
            supports_temperature = False
        elif model_name.startswith("o1") or model_name.startswith("o1-"):
            # O1 시리즈: max_completion_tokens 사용, temperature 미지원
            # 예: o1, o1-preview, o1-mini
            use_max_completion_tokens = True
            supports_temperature = False
        elif model_name.startswith("o3") or model_name.startswith("o3-"):
            # O3 시리즈: max_completion_tokens 사용, temperature 미지원
            # 예: o3, o3-mini
            use_max_completion_tokens = True
            supports_temperature = False
        elif model_name.startswith("gpt-4") or model_name.startswith("gpt-4-"):
            # GPT-4 시리즈 (구버전): max_tokens 사용, temperature 지원
            # 예: gpt-4, gpt-4-turbo, gpt-4-32k
            use_max_completion_tokens = False
            supports_temperature = True
        elif model_name.startswith("gpt-3.5") or "gpt-3.5" in model_name:
            # GPT-3.5 시리즈: max_tokens 사용, temperature 지원
            # 예: gpt-3.5-turbo, gpt-3.5-turbo-16k
            use_max_completion_tokens = False
            supports_temperature = True
        else:
            # 알 수 없는 모델: 기본값으로 max_completion_tokens 시도
            use_max_completion_tokens = True
            supports_temperature = True
        
        # OpenAI 공식 API 호출 파라미터 구성
        # 참고: https://platform.openai.com/docs/api-reference/chat/create
        create_kwargs = {
            "model": model_name,
            "messages": messages,
        }
        
        # max_completion_tokens / max_tokens 설정
        if max_completion_tokens is not None:
            create_kwargs["max_completion_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            if use_max_completion_tokens:
                create_kwargs["max_completion_tokens"] = max_tokens
            else:
                create_kwargs["max_tokens"] = max_tokens
        
        # temperature 파라미터: 지원하는 모델에만 추가
        if temperature is not None and supports_temperature:
            create_kwargs["temperature"] = temperature

        # OpenAI API 호출 (공식 문서 예제 스타일)
        # 참고: https://platform.openai.com/docs/api-reference/chat/create
        
        # 디버깅: 요청 파라미터 로깅 (verbose 모드에서만)
        import os
        debug_mode = os.getenv("OPENAI_DEBUG", "false").lower() == "true"
        if debug_mode:
            import json
            debug_kwargs = {k: v for k, v in create_kwargs.items() if k != "messages"}
            debug_kwargs["messages_count"] = len(create_kwargs["messages"])
            debug_kwargs["messages_preview"] = str(create_kwargs["messages"])[:200]
            print(f"[DEBUG] OpenAI API 요청 파라미터: {json.dumps(debug_kwargs, indent=2, ensure_ascii=False)}")
        
        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except Exception as e:
            # 에러를 더 자세히 로깅하기 위해 원본 에러 정보 보존
            original_error = e
            # OpenAI API 에러 처리 (공식 문서 기준)
            # 에러 정보 추출
            error_str = str(e)
            error_str_lower = error_str.lower()
            
            # OpenAI 에러 객체의 속성 확인 (openai.APIError 또는 openai.APIStatusError)
            error_body = getattr(e, 'body', None)
            error_code = getattr(e, 'code', None)
            error_response = getattr(e, 'response', None)
            
            # 에러 정보 파싱 (OpenAI API 에러 객체 구조 확인)
            error_info = {}
            
            # 1. error_body가 dict인 경우
            if error_body and isinstance(error_body, dict):
                error_info = error_body.get('error', {})
            
            # 2. error_response가 있는 경우 (httpx.Response)
            elif error_response:
                # json() 메서드로 읽기
                if hasattr(error_response, 'json'):
                    try:
                        error_data = error_response.json()
                        if isinstance(error_data, dict):
                            error_info = error_data.get('error', {})
                    except:
                        pass
                # text 속성으로 읽기
                elif hasattr(error_response, 'text'):
                    try:
                        import json
                        error_data = json.loads(error_response.text)
                        if isinstance(error_data, dict):
                            error_info = error_data.get('error', {})
                    except:
                        pass
                # read() 메서드로 읽기
                elif hasattr(error_response, 'read'):
                    try:
                        import json
                        error_data = json.loads(error_response.read().decode('utf-8'))
                        if isinstance(error_data, dict):
                            error_info = error_data.get('error', {})
                    except:
                        pass
            
            # 3. 에러 문자열에서 직접 파싱 (마지막 수단)
            if not error_info and 'error' in error_str_lower:
                # 에러 문자열에서 JSON 추출 시도
                try:
                    import json
                    import re
                    # JSON 패턴 찾기
                    json_match = re.search(r'\{[^{}]*"error"[^{}]*\}', error_str)
                    if json_match:
                        error_data = json.loads(json_match.group())
                        if isinstance(error_data, dict):
                            error_info = error_data.get('error', {})
                except:
                    pass
            
            error_param = error_info.get('param', '')
            error_code_val = error_info.get('code', '') or error_code or ''
            error_type = error_info.get('type', '')
            
            # 1. max_tokens 에러 처리 (unsupported_parameter)
            is_max_tokens_error = (
                (error_code_val == 'unsupported_parameter' and error_param == 'max_tokens') or
                (error_type == 'invalid_request_error' and 'max_tokens' in error_str_lower and 
                 'max_completion_tokens' in error_str_lower)
            )
            
            # 2. temperature 에러 처리 (unsupported_value 또는 unsupported_parameter)
            is_temperature_error = (
                (error_code_val == 'unsupported_value' and error_param == 'temperature') or
                (error_code_val == 'unsupported_parameter' and error_param == 'temperature') or
                (error_type == 'invalid_request_error' and 'temperature' in error_str_lower and 
                 ('unsupported' in error_str_lower or 'not support' in error_str_lower))
            )
            
            # 파라미터 자동 조정 및 재시도
            retry_needed = False
            
            if is_max_tokens_error and "max_tokens" in create_kwargs:
                # max_tokens를 max_completion_tokens로 변환
                max_comp_tokens = create_kwargs.pop("max_tokens")
                create_kwargs["max_completion_tokens"] = max_comp_tokens
                retry_needed = True
            
            if is_temperature_error and "temperature" in create_kwargs:
                # temperature 제거 (모델이 지원하지 않음)
                create_kwargs.pop("temperature")
                retry_needed = True
            
            # 재시도
            if retry_needed:
                try:
                    response = self.client.chat.completions.create(**create_kwargs)
                except Exception as retry_e:
                    # 재시도도 실패하면 원래 에러를 다시 발생
                    raise original_error from retry_e
            else:
                # 처리할 수 없는 에러는 그대로 발생
                raise original_error
        
        # 응답 검증 및 반환
        if not response or not response.choices:
            raise ValueError("OpenAI API 응답이 비어있습니다.")
        
        choice = response.choices[0]
        message = choice.message
        
        # finish_reason 확인 (GPT 5.1 등 일부 모델에서 중요)
        finish_reason = getattr(choice, 'finish_reason', None)
        if finish_reason and finish_reason not in ['stop', 'length']:
            # stop이나 length가 아니면 경고 (예: content_filter, null 등)
            import warnings
            warnings.warn(f"OpenAI API 응답의 finish_reason이 예상과 다릅니다: {finish_reason}")
        
        message_content = message.content
        if message_content is None:
            # GPT 5.1 등 일부 모델은 content가 None일 수 있음
            # 이 경우 전체 응답 구조를 확인
            raise ValueError(
                f"OpenAI API 응답의 content가 None입니다. "
                f"finish_reason: {finish_reason}, "
                f"response_id: {getattr(response, 'id', 'N/A')}"
            )
        
        # 빈 문자열 체크 및 디버깅 정보
        content_stripped = message_content.strip()
        if not content_stripped:
            # 빈 문자열인 경우 상세 정보 로깅
            import logging
            import os
            logger = logging.getLogger(__name__)
            debug_info = (
                f"OpenAI API 응답이 빈 문자열입니다. "
                f"model: {model_name}, "
                f"finish_reason: {finish_reason}, "
                f"original_length: {len(message_content)}, "
                f"response_id: {getattr(response, 'id', 'N/A')}, "
                f"usage: {getattr(response, 'usage', 'N/A')}"
            )
            logger.warning(debug_info)
            
            # 디버그 모드에서 전체 응답 출력
            debug_mode = os.getenv("OPENAI_DEBUG", "false").lower() == "true"
            if debug_mode:
                print(f"[DEBUG] {debug_info}")
                print(f"[DEBUG] 전체 응답 구조: {response}")
                if hasattr(response, 'choices') and response.choices:
                    print(f"[DEBUG] Choice 0: {response.choices[0]}")
                    print(f"[DEBUG] Message: {response.choices[0].message}")
                    print(f"[DEBUG] Message content (raw): {repr(message_content)}")
            
            # 빈 문자열이라도 반환 (호출자는 빈 문자열을 처리할 수 있어야 함)
        
        return content_stripped


