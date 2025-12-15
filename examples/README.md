# VLM 래퍼 예제

이 디렉토리에는 VLMManager 사용 예제가 포함되어 있습니다.

## 파일 설명

- `simple_example.py`: VLMManager의 기본적인 사용법을 보여주는 간단한 예제
- `test_vlm_wrapper.py`: VLMManager의 다양한 사용법을 보여주는 상세한 테스트 예제

## 사용 방법

### 1. 환경 변수 설정

`.env` 파일에 API 키를 설정하거나 환경 변수로 설정하세요:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="your-gemini-key"
export LOCAL_VLM_MODEL="Qwen/Qwen-VL-Chat"
```

### 2. 예제 실행

```bash
# 간단한 예제 실행
python examples/simple_example.py

# 상세한 테스트 실행
python examples/test_vlm_wrapper.py
```

### 3. 개별 테스트 실행

Python 인터프리터에서:

```python
from examples.test_vlm_wrapper import test_basic_usage
test_basic_usage()
```

## 주요 테스트 항목

1. **기본 사용법**: 단순 텍스트 프롬프트로 VLM 호출
2. **이미지 포함**: 이미지와 함께 VLM 호출
3. **구조화된 프롬프트**: system_prompt, context, user_prompt 사용
4. **여러 VLM 타입**: OpenAI, Gemini, Local 모델 테스트
5. **여러 이미지**: 여러 이미지를 동시에 처리
6. **추가 옵션**: max_tokens, max_new_tokens 등 옵션 사용
7. **에러 핸들링**: 잘못된 입력에 대한 오류 처리
8. **싱글톤 패턴**: VLMManager의 싱글톤 동작 확인
9. **하위 호환성**: 기존 vlm_utils.py 함수 사용

## 주의사항

- 테스트 이미지 파일이 필요한 경우, `test_image.jpg` 등의 경로를 실제 이미지 경로로 변경하세요.
- API 키가 설정되지 않은 경우 해당 테스트는 오류를 발생시킵니다.
- Local 모델 테스트는 모델이 다운로드되어 있어야 합니다.

