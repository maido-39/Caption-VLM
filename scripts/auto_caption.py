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
    - 출력: 입력 폴더 내 captions.json에 step(파일명), caption, context를 기록
"""
import argparse
import json
import sys
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


def get_vlm_kwargs(vlm: str) -> Dict[str, Any]:
    if vlm == "openai":
        return {"api_key": Settings.get_openai_key()}
    if vlm == "gemini":
        return {"api_key": Settings.get_gemini_key()}
    if vlm == "local":
        return {"model_name": Settings.get_local_model()}
    return {}


def process(
    images: List[Path],
    vlm: str,
    system_prompt: str,
    instruction: str,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    manager = VLMManager()
    context: str = ""
    results: List[Dict[str, Any]] = []

    for img_path in images:
        if verbose:
            print(f"[INFO] 처리 중: {img_path.name}")

        try:
            pil_img: Image.Image = convert_to_pil_image(str(img_path))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            prompt = build_prompt(system_prompt, context, instruction)

            vlm_kwargs = get_vlm_kwargs(vlm)
            caption = manager.call_vlm(
                vlm_name=vlm,
                images=[pil_img],
                prompt=prompt,
                **vlm_kwargs,
            )

            results.append(
                {
                    "step": img_path.name,
                    "caption": caption,
                    "context": context,
                }
            )

            # 다음 iteration의 context는 직전 캡션만 사용 (누적 없음)
            context = caption

            if verbose:
                print(f"[OK] {img_path.name}: {caption[:80]}...")
        except Exception as exc:  # pragma: no cover - 런타임 로깅 목적
            err_msg = f"오류 발생 ({img_path.name}): {exc}"
            results.append(
                {
                    "step": img_path.name,
                    "caption": None,
                    "context": context,
                    "error": err_msg,
                }
            )
            if verbose:
                print(f"[ERROR] {err_msg}")

    return results


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir).resolve()
    paths = ensure_paths(episode_dir)

    system_prompt = load_default_prompt(paths["prompt_path"])
    instruction = load_instruction(paths["episode_info"])
    images = pick_images(paths["rgb_dir"])

    if args.verbose:
        print(f"[INFO] 선택된 이미지 수: {len(images)} (마지막 포함)")
        print(f"[INFO] Instruction: {instruction}")

    results = process(
        images=images,
        vlm=args.vlm,
        system_prompt=system_prompt,
        instruction=instruction,
        verbose=args.verbose,
    )

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (episode_dir / "captions.json").resolve()
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 캡션이 저장되었습니다: {output_path}")


if __name__ == "__main__":
    main()

