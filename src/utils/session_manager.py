# -*- coding: utf-8 -*-
"""세션 관리 및 문서 기록 유틸리티"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from PIL import Image
import shutil

from src.vlm import VLMManager


class SessionManager:
    """세션 관리 클래스"""
    
    def __init__(self, export_base_dir: str = "export_docs"):
        """
        Args:
            export_base_dir: export 문서를 저장할 기본 디렉토리
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.export_base_dir = self.project_root / export_base_dir
        self.export_base_dir.mkdir(exist_ok=True)
        
        self.current_session_dir: Optional[Path] = None
        self.current_iteration: int = 0
        self.is_active: bool = False
        self.session_name: Optional[str] = None
    
    def start_session(
        self,
        context: str,
        vlm_type: str,
        api_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        local_model: Optional[str] = None
    ) -> str:
        """
        새 세션 시작
        
        Args:
            context: 첫 번째 컨텍스트 (폴더 이름 생성에 사용)
            vlm_type: VLM 타입
            api_key: OpenAI API 키
            gemini_key: Gemini API 키
            local_model: 로컬 모델 이름
        
        Returns:
            생성된 세션 폴더 경로
        """
        if self.is_active:
            return f"이미 활성화된 세션이 있습니다: {self.current_session_dir}"
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # AI를 사용하여 고유 폴더 이름 생성
        folder_name = self._generate_folder_name(
            context=context,
            vlm_type=vlm_type,
            api_key=api_key,
            gemini_key=gemini_key,
            local_model=local_model
        )
        
        # 폴더 이름 정리 (파일 시스템에 안전한 문자만 사용)
        folder_name = self._sanitize_folder_name(folder_name)
        
        # 전체 폴더 이름: timestamp_folder_name
        full_folder_name = f"{timestamp}_{folder_name}"
        
        # 세션 디렉토리 생성
        self.current_session_dir = self.export_base_dir / full_folder_name
        self.current_session_dir.mkdir(exist_ok=True)
        
        # 이미지 저장 디렉토리 생성
        imgs_dir = self.current_session_dir / "imgs"
        imgs_dir.mkdir(exist_ok=True)
        
        # Markdown 문서 초기화
        self._init_markdown_document()
        
        self.current_iteration = 0
        self.is_active = True
        self.session_name = full_folder_name
        
        return f"세션이 시작되었습니다: {full_folder_name}"
    
    def _generate_folder_name(
        self,
        context: str,
        vlm_type: str,
        api_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        local_model: Optional[str] = None
    ) -> str:
        """
        AI를 사용하여 컨텍스트에 맞는 폴더 이름 생성
        
        Args:
            context: 컨텍스트
            vlm_type: VLM 타입
            api_key: OpenAI API 키
            gemini_key: Gemini API 키
            local_model: 로컬 모델 이름
        
        Returns:
            생성된 폴더 이름
        """
        prompt = f"""다음 컨텍스트를 기반으로 짧고 명확한 영어 폴더 이름을 생성해주세요. 
폴더 이름은 파일 시스템에 안전해야 하므로 공백, 특수문자 없이 영문자, 숫자, 하이픈(-), 언더스코어(_)만 사용하세요.
최대 30자 이내로 간결하게 만들어주세요.

컨텍스트: {context}

폴더 이름만 출력하세요 (설명 없이):"""
        
        try:
            # VLMManager 사용
            manager = VLMManager()
            
            # 선택된 VLM 사용
            if vlm_type == "openai":
                from src.config.settings import Settings
                key = api_key or Settings.get_openai_key()
                if not key:
                    # API 키가 없으면 기본 이름 사용
                    return "session"
                folder_name = manager.call_vlm(
                    vlm_name="openai",
                    images=None,
                    prompt=prompt,
                    api_key=key,
                    max_tokens=50
                ).strip()
            elif vlm_type == "gemini":
                from src.config.settings import Settings
                key = gemini_key or Settings.get_gemini_key()
                if not key:
                    return "session"
                folder_name = manager.call_vlm(
                    vlm_name="gemini",
                    images=None,
                    prompt=prompt,
                    api_key=key
                ).strip()
            else:
                # Local 모델은 폴더 이름 생성에 사용하지 않음 (너무 느림)
                folder_name = "session"
            
            # 결과 정리
            folder_name = folder_name.replace("\n", "").replace(" ", "_").strip()
            if not folder_name or len(folder_name) > 50:
                folder_name = "session"
            
            return folder_name
        except Exception as e:
            print(f"폴더 이름 생성 중 오류 발생: {e}")
            return "session"
    
    def _sanitize_folder_name(self, name: str) -> str:
        """
        폴더 이름을 파일 시스템에 안전한 형태로 변환
        
        Args:
            name: 원본 이름
        
        Returns:
            정리된 이름
        """
        # 특수문자 제거, 공백을 언더스코어로
        import re
        name = re.sub(r'[^\w\-_]', '', name)
        name = name.replace(' ', '_')
        # 길이 제한
        if len(name) > 50:
            name = name[:50]
        if not name:
            name = "session"
        return name
    
    def _init_markdown_document(self):
        """Markdown 문서 초기화"""
        if not self.current_session_dir:
            return
        
        md_file = self.current_session_dir / "document.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# 세션 문서\n\n")
            f.write(f"**세션 시작 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
    
    def add_iteration(
        self,
        images: Optional[List[Image.Image]],
        caption: str,
        context: str
    ):
        """
        새로운 iteration 추가
        
        Args:
            images: 이미지 리스트
            caption: 생성된 캡션
            context: 사용된 컨텍스트
        """
        if not self.is_active or not self.current_session_dir:
            raise ValueError("활성화된 세션이 없습니다. 먼저 세션을 시작하세요.")
        
        self.current_iteration += 1
        md_file = self.current_session_dir / "document.md"
        imgs_dir = self.current_session_dir / "imgs"
        
        # Markdown에 iteration 추가
        with open(md_file, 'a', encoding='utf-8') as f:
            f.write(f"## Iteration {self.current_iteration}\n\n")
            f.write(f"**시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**컨텍스트**: {context}\n\n")
            
            # 이미지 저장 및 참조
            if images and len(images) > 0:
                f.write(f"### 이미지\n\n")
                for idx, img in enumerate(images):
                    # 이미지 파일명 생성: 원본이름_고유폴더명_timestamp.jpg
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    img_filename = f"img_{idx+1}_{self.session_name}_{timestamp}.jpg"
                    img_path = imgs_dir / img_filename
                    
                    # 이미지 저장
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(img_path, 'JPEG', quality=95)
                    
                    # Markdown에 이미지 참조 추가
                    relative_path = f"imgs/{img_filename}"
                    f.write(f"![이미지 {idx+1}]({relative_path})\n\n")
            
            # 캡션 추가
            f.write(f"### 캡션\n\n")
            f.write(f"{caption}\n\n")
            f.write(f"---\n\n")
    
    def save_session(self) -> str:
        """
        현재 세션 저장 (Markdown 문서 저장 완료)
        
        Returns:
            저장 메시지
        """
        if not self.is_active or not self.current_session_dir:
            return "저장할 활성 세션이 없습니다."
        
        md_file = self.current_session_dir / "document.md"
        with open(md_file, 'a', encoding='utf-8') as f:
            f.write(f"\n**세션 종료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**총 Iteration 수**: {self.current_iteration}\n\n")
        
        return f"세션이 저장되었습니다: {self.current_session_dir}"
    
    def end_session(self) -> str:
        """
        세션 종료
        
        Returns:
            종료 메시지
        """
        if not self.is_active:
            return "활성화된 세션이 없습니다."
        
        # 세션 저장
        self.save_session()
        
        session_name = self.session_name
        self.current_session_dir = None
        self.current_iteration = 0
        self.is_active = False
        self.session_name = None
        
        return f"세션이 종료되었습니다: {session_name}"
    
    def export_to_pdf(self) -> Optional[str]:
        """
        현재 세션을 PDF로 내보내기
        
        Returns:
            PDF 파일 경로 또는 None
        """
        if not self.is_active or not self.current_session_dir:
            return None
        
        try:
            # markdown2pdf 또는 다른 라이브러리 사용
            # 여기서는 markdown을 HTML로 변환 후 weasyprint로 PDF 생성
            # 또는 reportlab 사용
            
            # 간단한 방법: markdown을 HTML로 변환 후 weasyprint 사용
            # 또는 markdown-pdf 라이브러리 사용
            
            # 먼저 markdown 파일 읽기
            md_file = self.current_session_dir / "document.md"
            if not md_file.exists():
                return None
            
            # markdown을 HTML로 변환
            import markdown
            
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            html_content = markdown.markdown(
                md_content,
                extensions=['codehilite', 'fenced_code', 'tables', 'nl2br']
            )
            
            # HTML 템플릿에 삽입
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        line-height: 1.6;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        margin: 10px 0;
                    }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #555; margin-top: 30px; }}
                    h3 {{ color: #777; margin-top: 20px; }}
                    code {{
                        background-color: #f4f4f4;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }}
                    pre {{
                        background-color: #f4f4f4;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # 이미지 경로를 절대 경로로 변환
            html_template = html_template.replace(
                'src="imgs/',
                f'src="{self.current_session_dir}/imgs/'
            )
            
            # HTML을 PDF로 변환
            try:
                from weasyprint import HTML
                pdf_path = self.current_session_dir / "document.pdf"
                HTML(string=html_template, base_url=str(self.current_session_dir)).write_pdf(pdf_path)
                return str(pdf_path)
            except ImportError:
                # weasyprint가 없으면 reportlab 사용
                try:
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.lib.units import inch
                    from reportlab.pdfbase import pdfmetrics
                    from reportlab.pdfbase.ttfonts import TTFont
                    import re
                    
                    pdf_path = self.current_session_dir / "document.pdf"
                    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Markdown을 간단히 파싱하여 PDF 생성
                    lines = md_content.split('\n')
                    for line in lines:
                        if line.startswith('# '):
                            story.append(Paragraph(line[2:], styles['Title']))
                        elif line.startswith('## '):
                            story.append(Paragraph(line[3:], styles['Heading1']))
                        elif line.startswith('### '):
                            story.append(Paragraph(line[4:], styles['Heading2']))
                        elif line.startswith('!['):
                            # 이미지 처리
                            match = re.search(r'!\[.*?\]\((.*?)\)', line)
                            if match:
                                img_path = self.current_session_dir / match.group(1)
                                if img_path.exists():
                                    try:
                                        img = RLImage(str(img_path), width=5*inch, height=5*inch)
                                        story.append(img)
                                    except:
                                        pass
                        elif line.strip():
                            story.append(Paragraph(line, styles['Normal']))
                        else:
                            story.append(Spacer(1, 0.2*inch))
                    
                    doc.build(story)
                    return str(pdf_path)
                except ImportError:
                    return None
        except Exception as e:
            print(f"PDF 생성 중 오류 발생: {e}")
            return None

