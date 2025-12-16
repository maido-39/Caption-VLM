# VLM 이미지 캡션 생성 GUI 프로그램

Vision Language Model (VLM)을 사용하여 이미지에 대한 캡션을 생성하는 Gradio 기반 GUI 프로그램입니다.

## 기능

- **다중 VLM 지원**: OpenAI API, Gemini API, 로컬 VLM (Qwen 등) 중 선택 가능
- **유연한 이미지 입력**: 0~5개 이미지 입력 가능 (이미지 없이도 프롬프트만으로 가능)
- **커스텀 프롬프트**: 사용자가 원하는 프롬프트로 캡션 생성
- **API 키 관리**: 환경 변수 또는 UI를 통한 안전한 키 관리
- **Conda 환경 지원**: 재현 가능한 환경 설정

## 설치

### 사전 요구사항

- Python 3.10 이상 (3.10, 3.11, 3.12 권장)
- Conda 또는 pip 패키지 관리자

### 방법 1: Conda 환경 사용 (권장)

1. **Conda가 설치되어 있지 않은 경우:**
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 [Anaconda](https://www.anaconda.com/download) 설치

2. **환경 생성 및 활성화:**
   ```bash
   # 프로젝트 디렉토리로 이동
   cd caption-vlm
   
   # Conda 환경 생성 (Python 3.10 포함)
   conda env create -f environment.yml
   
   # 환경 활성화
   conda activate caption-vlm
   ```

3. **환경 확인:**
   ```bash
   python --version  # Python 3.10.x 확인
   ```

### 방법 2: pip + venv 사용

1. **Python 가상환경 생성:**
   ```bash
   # Python 3.10 이상이 설치되어 있는지 확인
   python --version
   
   # 가상환경 생성
   python -m venv venv
   
   # 가상환경 활성화
   # Linux/Mac:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

2. **패키지 설치:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 방법 3: pip만 사용 (시스템 Python)

```bash
# Python 3.10 이상이 설치되어 있는지 확인
python --version

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 설치 확인

설치가 완료되면 다음 명령어로 확인할 수 있습니다:

```bash
python -c "import gradio; import openai; import google.generativeai; print('모든 패키지가 정상적으로 설치되었습니다!')"
```

## 설정

### 환경 변수 설정

API 키를 환경 변수로 설정하면 UI에서 매번 입력할 필요가 없습니다.

1. **`.env` 파일 생성:**
   ```bash
   # .env.example을 복사하여 .env 파일 생성
   cp .env.example .env
   
   # 또는 직접 생성
   touch .env
   ```

2. **`.env` 파일 편집:**
   ```bash
   # 텍스트 에디터로 .env 파일 열기
   nano .env  # 또는 vim, code 등
   ```

3. **`.env` 파일 내용 예시:**
   ```env
   # OpenAI API Key (OpenAI VLM 사용 시 필요)
   OPENAI_API_KEY=sk-your-openai-api-key-here
   
   # OpenAI 모델 설정
   OPENAI_MODEL=gpt-4o
   OPENAI_TEMPERATURE=1.0
   
   # OpenAI 토큰 제한 설정
   # Reasoning 모델(gpt-5, o1, o3)용: max_completion_tokens
   # 기본값: 1000 (reasoning 최대 500 + 응답 최대 500)
   OPENAI_MAX_COMPLETION_TOKENS=1000
   
   # 일반 모델(gpt-4o, gpt-4 등)용: max_tokens
   # 기본값: 500
   OPENAI_MAX_TOKENS=500
   
   # Gemini API Key (Gemini VLM 사용 시 필요)
   GEMINI_API_KEY=your-gemini-api-key-here
   GEMINI_MODEL=gemini-2.0-flash-exp
   
   # 로컬 VLM 모델 이름 (로컬 VLM 사용 시)
   # 선택지 예시:
   # - Qwen/Qwen-VL-Chat (기본, 7B, ~14GB VRAM)
   # - Qwen/Qwen2-VL-2B-Instruct (경량, 2B, ~4GB VRAM)
   # - Qwen/Qwen2-VL-7B-Instruct (최신, 7B, ~14GB VRAM)
   # - llava-hf/llava-1.5-7b-hf (7B, ~14GB VRAM)
   # - Salesforce/blip2-opt-2.7b (경량, 2.7B, ~6GB VRAM)
   # - Salesforce/instructblip-vicuna-7b (7B, ~14GB VRAM)
   LOCAL_VLM_MODEL=Qwen/Qwen-VL-Chat
   ```

4. **API 키 발급 방법:**
   - **OpenAI API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)에서 발급
   - **Gemini API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)에서 발급
   - **로컬 VLM**: Hugging Face에서 모델 다운로드 (자동)

**참고:** 
- `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다
- UI에서도 API 키를 직접 입력할 수 있습니다 (환경 변수 우선)

## 사용 방법

### 프로그램 실행

```bash
# Conda 환경이 활성화되어 있는지 확인
conda activate caption-vlm

# Python 버전 확인 (3.10 이상이어야 함)
python --version  # 또는 python3 --version

# 프로그램 실행
python src/main.py
# 또는
python3 src/main.py
```

**참고:** 
- 시스템의 기본 `python`이 Python 2.x인 경우 `python3`를 사용하세요
- Conda 환경이 활성화되어 있으면 `python` 명령어가 올바른 버전을 사용합니다

브라우저에서 `http://localhost:7860`으로 접속하세요.

### 사용 순서

1. 이미지를 업로드하세요 (0~5개, 선택사항)
2. 사용할 VLM을 선택하세요 (OpenAI, Gemini, Local)
3. 프롬프트를 입력하세요
4. 필요시 API 키를 입력하세요 (환경 변수에 설정되어 있으면 생략 가능)
5. "캡션 생성" 버튼을 클릭하세요

## 프로젝트 구조

```
caption-vlm/
├── src/
│   ├── main.py                 # Gradio 앱 진입점
│   ├── vlm_utils.py            # VLM 통합 유틸리티
│   ├── config/
│   │   └── settings.py         # 설정 관리
│   └── utils/
│       └── image_utils.py      # 이미지 처리 유틸리티
├── requirements.txt
├── environment.yml
├── .env.example
└── README.md
```

## 지원하는 VLM

### API 기반 VLM

- **OpenAI**: GPT-4o Vision 모델 사용
- **Gemini**: Gemini 2.0 Flash 모델 사용

### 로컬 VLM 모델 선택지

로컬 VLM을 사용하려면 `.env` 파일의 `LOCAL_VLM_MODEL`에 모델 이름을 설정하거나, UI에서 직접 입력할 수 있습니다.

#### Qwen-VL 시리즈 (권장)

| 모델 이름 | 파라미터 | 메모리 요구사항 | 특징 |
|---------|---------|---------------|------|
| `Qwen/Qwen-VL-Chat` | 7B | ~14GB VRAM | 기본 모델, 안정적 |
| `Qwen/Qwen-VL` | 7B | ~14GB VRAM | Chat 버전 없이 순수 모델 |
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | ~4GB VRAM | 경량 모델, 빠른 추론 |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | ~14GB VRAM | 최신 버전, 향상된 성능 |

#### LLaVA 시리즈

| 모델 이름 | 파라미터 | 메모리 요구사항 | 특징 |
|---------|---------|---------------|------|
| `llava-hf/llava-1.5-7b-hf` | 7B | ~14GB VRAM | 널리 사용되는 VLM |
| `llava-hf/llava-1.5-13b-hf` | 13B | ~26GB VRAM | 더 큰 모델, 향상된 성능 |

#### BLIP-2 시리즈

| 모델 이름 | 파라미터 | 메모리 요구사항 | 특징 |
|---------|---------|---------------|------|
| `Salesforce/blip2-opt-2.7b` | 2.7B | ~6GB VRAM | 경량 모델 |
| `Salesforce/blip2-opt-6.7b` | 6.7B | ~14GB VRAM | 중간 크기 모델 |

#### InstructBLIP

| 모델 이름 | 파라미터 | 메모리 요구사항 | 특징 |
|---------|---------|---------------|------|
| `Salesforce/instructblip-vicuna-7b` | 7B | ~14GB VRAM | 지시사항 따르기에 최적화 |

#### CogVLM

| 모델 이름 | 파라미터 | 메모리 요구사항 | 특징 |
|---------|---------|---------------|------|
| `THUDM/cogvlm-chat-hf` | 17B | ~34GB VRAM | 대규모 모델, 높은 성능 |

**참고:**
- 모델은 Hugging Face에서 자동으로 다운로드됩니다 (최초 실행 시)
- GPU가 없어도 CPU에서 실행 가능하지만 매우 느립니다
- VRAM 요구사항은 대략적인 값이며, 실제 사용량은 다를 수 있습니다
- 모델 이름은 Hugging Face Hub의 정확한 모델 ID를 사용해야 합니다

## CLI 스크립트: 자동 캡션 생성

프로젝트에는 배치로 이미지에 캡션을 생성하는 CLI 스크립트가 포함되어 있습니다.

### 사용법

```bash
# 기본 사용법
python scripts/auto_caption.py sample_data/ep_0000_00508-4vwGX7U38Ux_wardrobe

# 출력 파일 지정
python scripts/auto_caption.py sample_data/ep_0000_00508-4vwGX7U38Ux_wardrobe --output custom_captions.json

# VLM 선택 (openai, gemini, local)
python scripts/auto_caption.py sample_data/ep_0000_00508-4vwGX7U38Ux_wardrobe --vlm openai

# Temperature 설정
python scripts/auto_caption.py sample_data/ep_0000_00508-4vwGX7U38Ux_wardrobe --temperature 0.1

# 상세 로그 출력
python scripts/auto_caption.py sample_data/ep_0000_00508-4vwGX7U38Ux_wardrobe --verbose
```

### 기능

- **이미지 선택**: 4프레임 간격으로 이미지 선택 (0, 4, 8, ...), 마지막 이미지는 항상 포함
- **기본 프롬프트**: 프로젝트 루트의 `default_prompt.txt` 사용
- **Instruction**: `episode_info.json`의 `object_goal`에서 자동 추출 → `Find and move toward "<object_goal>"`
- **Context 관리**: 직전 캡션을 다음 호출의 context로 전달 (누적 없음)
- **JSON 출력**: 각 이미지별 캡션, 토큰 사용량, 실행 시간 등 구조화된 정보 저장

### 출력 JSON 구조

```json
{
  "api": {
    "vlm": "openai",
    "model": "gpt-4o",
    "temperature": 1.0,
    "execution_times": [1.23, 1.45, ...],
    "total_execution_time": 10.5,
    "start_timestamp": "2025-12-15_21-33-36",
    "end_timestamp": "2025-12-15_21-33-46",
    "total_token_usage": {
      "prompt_tokens": 1500,
      "completion_tokens": 2000,
      "total_tokens": 3500,
      "reasoning_tokens": 500
    }
  },
  "dataset": {
    "target_path": "/path/to/episode_dir"
  },
  "captions": [
    {
      "step": "0000",
      "caption": "1. **Scene Description**: ...",
      "context": "",
      "token_usage": {
        "prompt_tokens": 500,
        "completion_tokens": 400,
        "total_tokens": 900,
        "reasoning_tokens": 200
      }
    }
  ]
}
```

### CaptionJsonParser 클래스

JSON 파일을 파싱하는 유틸리티 클래스가 포함되어 있습니다:

```python
from scripts.auto_caption import CaptionJsonParser

parser = CaptionJsonParser("captions.json")

# 각 섹션별 리스트 반환
descriptions = parser.get_description()
plannings = parser.get_planning()
summaries = parser.get_historical_summarization()
instructions = parser.get_immediate_action_instruction()

# Step 정보
steps_info = parser.get_steps()  # {"total_step": 10, "steps": ["0000", "0004", ...]}

# Step-Caption 딕셔너리
caption_dict = parser.get_caption()  # {"0000": "caption text", ...}
```

## OpenAI 모델별 설정

### 모델별 파라미터 지원

프로젝트는 모델별로 자동으로 적절한 파라미터를 설정합니다:

| 모델 시리즈 | Reasoning 지원 | 토큰 파라미터 | Temperature 지원 |
|------------|---------------|--------------|-----------------|
| gpt-4o | ❌ | max_completion_tokens | ✅ |
| gpt-4 | ❌ | max_tokens | ✅ |
| gpt-3.5 | ❌ | max_tokens | ✅ |
| gpt-5 | ✅ | max_completion_tokens | ❌ |
| o1 | ✅ | max_completion_tokens | ❌ |
| o3 | ✅ | max_completion_tokens | ❌ |

### Reasoning 모델 주의사항

Reasoning 모델(gpt-5, o1, o3)은 reasoning 토큰과 실제 응답 토큰을 모두 포함하므로 더 큰 토큰 제한이 필요합니다:
- Reasoning 토큰: 최대 500 토큰
- 실제 응답 토큰: 설정한 값만큼
- 총 필요 토큰: reasoning + 응답

예를 들어, 응답을 400 토큰으로 제한하려면 `OPENAI_MAX_COMPLETION_TOKENS=900` (500 + 400)으로 설정해야 합니다.

### 모델별 비용 참고표

```
+------------------+------------------+------------------+------------------+
|      모델명      |  Reasoning 지원  |   상대적 비용    |   토큰 파라미터  |
+------------------+------------------+------------------+------------------+
| gpt-4            |        ❌        |      중간        |    max_tokens    |
| gpt-4-turbo      |        ❌        |      중간        |    max_tokens    |
| gpt-4o           |        ❌        |      낮음        | max_completion_  |
|                  |                  |                  |     tokens       |
| gpt-4o-mini      |        ❌        |      매우 낮음   | max_completion_  |
|                  |                  |                  |     tokens       |
| gpt-5            |        ✅        |      높음       | max_completion_  |
| gpt-5.1          |        ✅        |      높음       | max_completion_  |
| gpt-5-mini       |        ✅        |      중간       | max_completion_  |
|                  |                  |                  |     tokens       |
| o1               |        ✅        |      높음       | max_completion_  |
| o1-preview       |        ✅        |      높음       | max_completion_  |
| o1-mini          |        ✅        |      중간       | max_completion_  |
|                  |                  |                  |     tokens       |
| o3               |        ✅        |      높음       | max_completion_  |
| o3-mini          |        ✅        |      중간       | max_completion_  |
|                  |                  |                  |     tokens       |
+------------------+------------------+------------------+------------------+
```

## 주의사항

- API 키는 `.env` 파일에 저장하거나 UI에서 입력할 수 있습니다
- 로컬 VLM을 사용하려면 충분한 GPU 메모리가 필요할 수 있습니다
- 이미지는 최대 5개까지 입력 가능합니다
- 이미지가 없어도 프롬프트만으로 텍스트 생성이 가능합니다
- Reasoning 모델 사용 시 토큰 제한을 충분히 설정해야 합니다 (기본값: 1000)
- 토큰 사용량은 JSON 파일에 자동으로 기록됩니다

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

# Caption-VLM
