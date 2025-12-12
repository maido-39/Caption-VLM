# -*- coding: utf-8 -*-
"""이미지 처리 유틸리티"""
import base64
import io
from typing import List, Optional, Union
from PIL import Image
import numpy as np


def validate_images(images: Optional[List]) -> bool:
    """
    이미지 리스트 검증 (0~5개 허용)
    
    Args:
        images: 이미지 리스트 (None 또는 최대 5개)
    
    Returns:
        bool: 유효한 경우 True
    """
    if images is None:
        return True
    
    if len(images) > 5:
        return False
    
    return True


def convert_to_pil_image(image: Union[str, Image.Image, bytes, tuple, np.ndarray]) -> Image.Image:
    """
    다양한 형식의 이미지를 PIL Image로 변환
    
    Args:
        image: 이미지 (경로, PIL Image, bytes, tuple, numpy array)
    
    Returns:
        PIL Image 객체
    """
    # PIL Image인 경우
    if isinstance(image, Image.Image):
        return image
    
    # tuple인 경우 (Gradio Gallery가 반환하는 형식일 수 있음)
    if isinstance(image, tuple):
        # tuple의 첫 번째 요소가 이미지일 가능성이 높음
        if len(image) > 0:
            return convert_to_pil_image(image[0])
        else:
            raise ValueError("빈 tuple은 이미지로 변환할 수 없습니다.")
    
    # numpy array인 경우
    if isinstance(image, np.ndarray):
        # uint8 타입으로 변환
        if image.dtype != np.uint8:
            # 0-1 범위의 float인 경우 0-255로 변환
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return Image.fromarray(image)
    
    # str인 경우 (파일 경로)
    if isinstance(image, str):
        return Image.open(image)
    
    # bytes인 경우
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image))
    
    raise ValueError(f"지원하지 않는 이미지 형식: {type(image)}")


def process_images(images: Optional[List]) -> Optional[List[Image.Image]]:
    """
    이미지 리스트를 PIL Image 리스트로 변환
    
    Args:
        images: 이미지 리스트 (None 허용)
    
    Returns:
        PIL Image 리스트 또는 None
    """
    if images is None or len(images) == 0:
        return None
    
    if not validate_images(images):
        raise ValueError("이미지는 최대 5개까지 허용됩니다.")
    
    processed = []
    for img in images:
        pil_image = convert_to_pil_image(img)
        processed.append(pil_image)
    
    return processed


def encode_image_base64(image: Image.Image) -> str:
    """
    PIL Image를 Base64 문자열로 인코딩
    
    Args:
        image: PIL Image 객체
    
    Returns:
        Base64 인코딩된 문자열
    """
    buffered = io.BytesIO()
    # RGB 모드로 변환 (일부 이미지는 RGBA 등일 수 있음)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    이미지 리사이징 (최대 크기 제한)
    
    Args:
        image: PIL Image 객체
        max_size: 최대 크기 (가로 또는 세로)
    
    Returns:
        리사이즈된 PIL Image 객체
    """
    width, height = image.size
    
    if width <= max_size and height <= max_size:
        return image
    
    # 비율 유지하며 리사이즈
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

