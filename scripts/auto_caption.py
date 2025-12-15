#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB 이미지들을 4프레임 간격으로 순회하며 VLM 캡션을 생성하는 CLI 스크립트.

사용 예:
    python scripts/auto_caption.py sample_data/ep_0000_00508-4vwGX7U38Ux_wardrobe

기능 요약:
    - 기본 프롬프트: 프로젝트 루트의 default_prompt.txt 사용
    - Instruction: episode_info.json의 object_goal → 'Find and move toward "<object_goal>"'
    - Context: 직전 캡션을 다음 호출의 context로 전달(누적 없음)
    - 이미지 선택: 0,4,8,... 번째 이미지 + 마지막 이미지를 반드시 포함
    - 출력: 입력 폴더 내 captions.json에 구조화된 정보 기록
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm import VLMManager
from src.utils.image_utils import convert_to_pil_image
from src.config.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB 이미지 자동 캡션 생성 CLI")
    parser.add_argument(
        "episode_dir",
        type=str,
        help="episode_info.json과 rgb/가 포함된 폴더 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 JSON 경로 (기본: <episode_dir>/captions.json)",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="openai",
        choices=["openai", "gemini", "local"],
        help="사용할 VLM 타입",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature 파라미터 (기본: None, 모델 기본값 사용)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    return parser.parse_args()


def ensure_paths(episode_dir: Path) -> Dict[str, Path]:
    rgb_dir = episode_dir / "rgb"
    episode_info = episode_dir / "episode_info.json"
    prompt_path = PROJECT_ROOT / "default_prompt.txt"

    missing = []
    if not episode_dir.is_dir():
        missing.append(f"폴더가 존재하지 않습니다: {episode_dir}")
    if not rgb_dir.is_dir():
        missing.append(f"rgb 폴더가 없습니다: {rgb_dir}")
    if not episode_info.is_file():
        missing.append(f"episode_info.json이 없습니다: {episode_info}")
    if not prompt_path.is_file():
        missing.append(f"default_prompt.txt가 없습니다: {prompt_path}")

    if missing:
        raise FileNotFoundError("\n".join(missing))

    return {
        "rgb_dir": rgb_dir,
        "episode_info": episode_info,
        "prompt_path": prompt_path,
    }


def load_default_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def load_instruction(episode_info_path: Path) -> str:
    data = json.loads(episode_info_path.read_text(encoding="utf-8"))
    object_goal = data.get("object_goal")
    if not object_goal:
        raise ValueError("episode_info.json에서 object_goal을 찾을 수 없습니다.")
    return f'Find and move toward "{object_goal}"'


def pick_images(rgb_dir: Path) -> List[Path]:
    image_paths = sorted(rgb_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"png 이미지가 없습니다: {rgb_dir}")

    selected: List[Path] = []
    for idx, img_path in enumerate(image_paths):
        if idx % 4 == 0:
            selected.append(img_path)

    # 마지막 이미지는 반드시 포함
    last_img = image_paths[-1]
    if last_img not in selected:
        selected.append(last_img)

    return selected


def build_prompt(system_prompt: str, context: str, instruction: str) -> Dict[str, str]:
    prompt: Dict[str, str] = {"system_prompt": system_prompt, "user_prompt": instruction}
    if context.strip():
        prompt["context"] = context.strip()
    return prompt


def get_vlm_kwargs(vlm: str, temperature: Optional[float] = None) -> Dict[str, Any]:
    """VLM kwargs 생성. temperature가 None이면 Settings에서 가져옴"""
    kwargs = {}
    if vlm == "openai":
        kwargs["api_key"] = Settings.get_openai_key()
        # temperature 우선순위: 인자 > Settings > None
        final_temperature = temperature if temperature is not None else Settings.get_openai_temperature()
        if final_temperature is not None:
            kwargs["temperature"] = final_temperature
    elif vlm == "gemini":
        kwargs["api_key"] = Settings.get_gemini_key()
        kwargs["model_name"] = Settings.get_gemini_model()  # 설정에서 모델 이름 가져오기
        if temperature is not None:
            kwargs["temperature"] = temperature
    elif vlm == "local":
        kwargs["model_name"] = Settings.get_local_model()
        if temperature is not None:
            kwargs["temperature"] = temperature
    return kwargs


def get_model_name(vlm: str) -> str:
    """VLM 타입에 따른 모델 이름 반환"""
    if vlm == "openai":
        return Settings.get_openai_model() or "gpt-4o"
    elif vlm == "gemini":
        return Settings.get_gemini_model() or "gemini-2.0-flash-exp"
    elif vlm == "local":
        return Settings.get_local_model()
    return "unknown"


def format_timestamp(dt: datetime) -> str:
    """YYYY-MM-DD_HH-MM-SS 형식으로 timestamp 반환"""
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def create_initial_json(
    output_path: Path,
    episode_dir: Path,
    vlm: str,
    model: str,
    temperature: Optional[float],
) -> None:
    """초기 JSON 파일 생성"""
    start_time = datetime.now()
    data = {
        "api": {
            "vlm": vlm,
            "model": model,
            "temperature": temperature,
            "execution_times": [],
            "total_execution_time": None,
            "start_timestamp": format_timestamp(start_time),
            "end_timestamp": None,
        },
        "dataset": {
            "target_path": str(episode_dir.resolve()),
        },
        "captions": [],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_json_caption(
    output_path: Path,
    step: str,
    caption: Optional[str],
    context: str,
    execution_time: float,
    error: Optional[str] = None,
) -> None:
    """JSON 파일에 캡션 하나 추가"""
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    caption_entry = {
        "step": step,
        "caption": caption,
        "context": context,
    }
    if error:
        caption_entry["error"] = error
    
    data["captions"].append(caption_entry)
    data["api"]["execution_times"].append(execution_time)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_json_final(
    output_path: Path,
    total_time: float,
) -> None:
    """JSON 파일에 최종 메타데이터 업데이트"""
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    end_time = datetime.now()
    data["api"]["total_execution_time"] = total_time
    data["api"]["end_timestamp"] = format_timestamp(end_time)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process(
    images: List[Path],
    vlm: str,
    system_prompt: str,
    instruction: str,
    output_path: Path,
    temperature: Optional[float] = None,
    verbose: bool = False,
) -> None:
    """이미지들을 처리하고 각각 JSON에 저장"""
    manager = VLMManager()
    context: str = ""

    for img_path in images:
        if verbose:
            print(f"[INFO] Processing: {img_path.name}")

        execution_time = 0.0
        caption = None
        error = None
        start_time = time.time()

        try:
            pil_img: Image.Image = convert_to_pil_image(str(img_path))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            prompt = build_prompt(system_prompt, context, instruction)

            vlm_kwargs = get_vlm_kwargs(vlm, temperature)
            
            # VLM 호출
            caption = manager.call_vlm(
                vlm_name=vlm,
                images=[pil_img],
                prompt=prompt,
                **vlm_kwargs,
            )
            execution_time = time.time() - start_time

            # 빈 문자열 체크 및 경고
            if not caption or not caption.strip():
                if verbose:
                    print(f"[WARNING] {img_path.name}: 빈 캡션이 반환되었습니다. (took {execution_time:.2f}s)")
                caption = ""  # 명시적으로 빈 문자열로 설정

            # 다음 iteration의 context는 직전 캡션만 사용 (누적 없음)
            # 빈 문자열이 아닐 때만 context 업데이트
            if caption and caption.strip():
                context = caption
            # 빈 문자열이면 이전 context 유지 (첫 번째 이미지면 빈 문자열)

            if verbose:
                if caption and caption.strip():
                    preview = caption[:80] if len(caption) > 80 else caption
                    print(f"[OK] {img_path.name}: {preview}... (took {execution_time:.2f}s)")
                else:
                    print(f"[OK] {img_path.name}: (빈 응답) (took {execution_time:.2f}s)")
        except Exception as exc:  # pragma: no cover - 런타임 로깅 목적
            execution_time = time.time() - start_time
            error = f"Error occurred ({img_path.name}): {exc}"
            if verbose:
                print(f"[ERROR] {error}")

        # 각 캡션을 즉시 JSON에 저장
        # caption이 None이면 context는 업데이트하지 않음 (이전 context 유지)
        current_context = context if caption else (context if context else "")
        # step은 숫자로만 저장 (예: "0000.png" -> "0000")
        step_number = img_path.stem  # .png 확장자 제거
        update_json_caption(
            output_path=output_path,
            step=step_number,
            caption=caption,
            context=current_context,
            execution_time=execution_time,
            error=error,
        )


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir).resolve()
    paths = ensure_paths(episode_dir)

    system_prompt = load_default_prompt(paths["prompt_path"])
    instruction = load_instruction(paths["episode_info"])
    images = pick_images(paths["rgb_dir"])

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (episode_dir / "captions.json").resolve()
    )

    # 모델 이름 가져오기
    model_name = get_model_name(args.vlm)
    
    # temperature 결정: 인자 > Settings > None
    final_temperature = args.temperature
    if final_temperature is None and args.vlm == "openai":
        final_temperature = Settings.get_openai_temperature()

    if args.verbose:
        print(f"[INFO] Selected images: {len(images)} (last included)")
        print(f"[INFO] Instruction: {instruction}")
        print(f"[INFO] VLM: {args.vlm}, Model: {model_name}")
        if final_temperature is not None:
            print(f"[INFO] Temperature: {final_temperature}")

    # 프로그램 시작 시 초기 JSON 파일 생성
    create_initial_json(
        output_path=output_path,
        episode_dir=episode_dir,
        vlm=args.vlm,
        model=model_name,
        temperature=final_temperature,
    )

    if args.verbose:
        print(f"[INFO] Initial JSON created: {output_path}")

    # 전체 실행 시간 측정
    total_start_time = time.time()

    # 이미지 처리 (각각 JSON에 저장)
    process(
        images=images,
        vlm=args.vlm,
        system_prompt=system_prompt,
        instruction=instruction,
        output_path=output_path,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # 전체 실행 시간 계산
    total_execution_time = time.time() - total_start_time

    # 최종 메타데이터 업데이트
    update_json_final(
        output_path=output_path,
        total_time=total_execution_time,
    )

    print(f"[DONE] Captions saved: {output_path}")
    print(f"[DONE] Total execution time: {total_execution_time:.2f}s")


class CaptionJsonParser:
    """JSON 캡션 파일 파서 클래스"""
    
    def __init__(self, json_path: str):
        """
        JSON 파일 경로로 초기화
        
        Args:
            json_path: JSON 파일 경로
        """
        self.json_path = Path(json_path)
        self.data = None
        self.load_json()
    
    def load_json(self) -> None:
        """JSON 파일 로드"""
        with self.json_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def _extract_section(self, caption_text: str, section_name: str) -> Optional[str]:
        """캡션 텍스트에서 특정 섹션 추출"""
        if not caption_text:
            return None
        
        # 섹션 패턴 찾기 (더 유연한 패턴)
        patterns = {
            "description": r"1\.\s*\*\*Scene Description\*\*:\s*(.*?)(?=\n\n2\.|\n\n\*\*\*END OF REASONING CONTEXT)",
            "planning": r"2\.\s*\*\*High-Level Planning\*\*:\s*(.*?)(?=\n\n3\.|\n\n\*\*\*END OF REASONING CONTEXT)",
            "historical_summarization": r"3\.\s*\*\*Historical Summarization\*\*:\s*(.*?)(?=\n\n4\.|\n\n\*\*\*END OF REASONING CONTEXT)",
            "immediate_action_instruction": r"4\.\s*\*\*Immediate Action Instruction\*\*:\s*(.*?)(?=\n\n\*\*\*END OF REASONING CONTEXT)",
        }
        
        pattern = patterns.get(section_name)
        if not pattern:
            return None
        
        match = re.search(pattern, caption_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def get_description(self) -> List[Optional[str]]:
        """Scene Description 섹션 리스트 반환"""
        descriptions = []
        for caption_entry in self.data.get("captions", []):
            caption_text = caption_entry.get("caption", "")
            desc = self._extract_section(caption_text, "description")
            descriptions.append(desc)
        return descriptions
    
    def get_planning(self) -> List[Optional[str]]:
        """High-Level Planning 섹션 리스트 반환"""
        plannings = []
        for caption_entry in self.data.get("captions", []):
            caption_text = caption_entry.get("caption", "")
            planning = self._extract_section(caption_text, "planning")
            plannings.append(planning)
        return plannings
    
    def get_historical_summarization(self) -> List[Optional[str]]:
        """Historical Summarization 섹션 리스트 반환"""
        summaries = []
        for caption_entry in self.data.get("captions", []):
            caption_text = caption_entry.get("caption", "")
            summary = self._extract_section(caption_text, "historical_summarization")
            summaries.append(summary)
        return summaries
    
    def get_immediate_action_instruction(self) -> List[Optional[str]]:
        """Immediate Action Instruction 섹션 리스트 반환"""
        instructions = []
        for caption_entry in self.data.get("captions", []):
            caption_text = caption_entry.get("caption", "")
            instruction = self._extract_section(caption_text, "immediate_action_instruction")
            instructions.append(instruction)
        return instructions
    
    def get_steps(self) -> Dict[str, Any]:
        """
        Step 정보 반환
        
        Returns:
            {
                "total_step": int,
                "steps": List[str]
            }
        """
        steps = []
        for caption_entry in self.data.get("captions", []):
            step = caption_entry.get("step", "")
            steps.append(step)
        
        return {
            "total_step": len(steps),
            "steps": steps,
        }
    
    def get_caption(self) -> Dict[str, str]:
        """
        Step-Caption 딕셔너리 반환
        
        Returns:
            {step: caption} 형태의 dict
        """
        caption_dict = {}
        for caption_entry in self.data.get("captions", []):
            step = caption_entry.get("step", "")
            caption = caption_entry.get("caption", "")
            caption_dict[step] = caption
        return caption_dict


if __name__ == "__main__":
    main()

