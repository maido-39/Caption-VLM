# -*- coding: utf-8 -*-
"""Gradio 기반 VLM 이미지 캡션 생성 GUI"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from typing import List, Optional, Tuple
from PIL import Image

from src.vlm_utils import generate_caption
from src.config.settings import Settings
from src.utils.session_manager import SessionManager

# 전역 세션 매니저
session_manager = SessionManager()


def create_caption(
    images: Optional[List],
    vlm_type: str,
    system_prompt: str,
    context: str,
    iteration_context: str,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    local_model: Optional[str] = None
) -> Tuple[str, str, Optional[List], str, str]:
    """
    캡션 생성 함수
    
    Args:
        images: 업로드된 이미지 리스트
        vlm_type: VLM 타입 ("openai", "gemini", "local")
        system_prompt: 시스템 프롬프트 (고정, 계속 사용)
        context: 질문/컨텍스트 (계속 유지됨)
        iteration_context: 이번 iteration에만 사용할 추가 컨텍스트 (초기화됨)
        openai_key: OpenAI API 키 (선택적)
        gemini_key: Gemini API 키 (선택적)
        local_model: 로컬 모델 이름 (선택적)
    
    Returns:
        (생성된 캡션, 유지할 context, 빈 이미지 리스트, 빈 iteration_context, 상태 메시지) 튜플
    """
    try:
        # 세션이 활성화되어 있는지 확인
        if not session_manager.is_active:
            return "오류: 세션이 시작되지 않았습니다. '프로젝트 시작' 버튼을 먼저 클릭하세요.", context, None, "", ""
        
        # 이미지 처리
        processed_images = None
        if images:
            # Gradio에서 받은 이미지를 PIL Image로 변환
            from src.utils.image_utils import convert_to_pil_image
            processed_images = [convert_to_pil_image(img) for img in images]
        
        # 프롬프트 결합: 시스템 프롬프트 + context + iteration_context
        prompt_parts = []
        if system_prompt.strip():
            prompt_parts.append(system_prompt.strip())
        if context.strip():
            prompt_parts.append(context.strip())
        if iteration_context.strip():
            prompt_parts.append(iteration_context.strip())
        
        if prompt_parts:
            prompt = "\n\n".join(prompt_parts)
        else:
            prompt = "이 이미지에 대해 자세히 설명해주세요."
        
        # API 키 설정 (UI에서 입력된 경우)
        api_key = None
        if vlm_type == "openai":
            api_key = openai_key or Settings.get_openai_key()
        elif vlm_type == "gemini":
            api_key = gemini_key or Settings.get_gemini_key()
        
        # 모델 이름 설정
        model_name = local_model or Settings.get_local_model()
        
        # 캡션 생성
        caption = generate_caption(
            vlm_type=vlm_type,
            images=processed_images,
            prompt=prompt,
            api_key=api_key,
            model_name=model_name if vlm_type == "local" else None
        )
        
        # 클립보드에 캡션 복사
        try:
            import pyperclip
            pyperclip.copy(caption)
        except ImportError:
            # pyperclip이 없으면 무시
            pass
        except Exception:
            # 클립보드 복사 실패해도 계속 진행
            pass
        
        # 세션에 iteration 추가 (context와 iteration_context 모두 기록)
        full_context = context
        if iteration_context.strip():
            full_context = f"{context}\n\n추가 컨텍스트: {iteration_context}" if context.strip() else iteration_context
        
        try:
            session_manager.add_iteration(
                images=processed_images,
                caption=caption,
                context=full_context
            )
            status_msg = f"Iteration {session_manager.current_iteration} 기록 완료 (클립보드에 복사됨)"
        except Exception as e:
            status_msg = f"기록 중 오류: {str(e)}"
        
        # 캡션, 유지할 context, 빈 이미지 리스트, 생성된 캡션(추가 컨텍스트에 복사), 상태 메시지 반환
        return caption, context, None, caption, status_msg
    
    except Exception as e:
        return f"오류 발생: {str(e)}", context, None, "", ""


def start_project(
    system_prompt: str,
    context: str,
    vlm_type: str,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    local_model: Optional[str] = None
) -> Tuple[str, str]:
    """
    프로젝트 시작
    
    Args:
        system_prompt: 시스템 프롬프트
        context: 질문/컨텍스트
        vlm_type: VLM 타입
        openai_key: OpenAI API 키
        gemini_key: Gemini API 키
        local_model: 로컬 모델 이름
    
    Returns:
        (상태 메시지, 세션 상태) 튜플
    """
    try:
        # 시스템 프롬프트와 컨텍스트를 결합하여 폴더 이름 생성에 사용
        combined_context = ""
        if system_prompt.strip() and context.strip():
            combined_context = f"{system_prompt.strip()}\n\n{context.strip()}"
        elif system_prompt.strip():
            combined_context = system_prompt.strip()
        elif context.strip():
            combined_context = context.strip()
        else:
            combined_context = "일반 이미지 분석 프로젝트"
        
        api_key = openai_key or Settings.get_openai_key() if vlm_type == "openai" else None
        gemini_key_val = gemini_key or Settings.get_gemini_key() if vlm_type == "gemini" else None
        local_model_val = local_model or Settings.get_local_model() if vlm_type == "local" else None
        
        msg = session_manager.start_session(
            context=combined_context,
            vlm_type=vlm_type,
            api_key=api_key,
            gemini_key=gemini_key_val,
            local_model=local_model_val
        )
        status = f"활성 세션: {session_manager.session_name}" if session_manager.is_active else "세션 없음"
        return msg, status
    except Exception as e:
        return f"오류: {str(e)}", "세션 없음"


def save_project() -> Tuple[str, str]:
    """
    프로젝트 저장
    
    Returns:
        (상태 메시지, 세션 상태) 튜플
    """
    try:
        msg = session_manager.save_session()
        status = f"활성 세션: {session_manager.session_name}" if session_manager.is_active else "세션 없음"
        return msg, status
    except Exception as e:
        return f"오류: {str(e)}", "세션 없음"


def end_project() -> Tuple[str, str]:
    """
    프로젝트 종료
    
    Returns:
        (상태 메시지, 세션 상태) 튜플
    """
    try:
        msg = session_manager.end_session()
        return msg, "세션 없음"
    except Exception as e:
        return f"오류: {str(e)}", "세션 없음"


def export_pdf() -> Tuple[str, Optional[str]]:
    """
    PDF로 내보내기
    
    Returns:
        (상태 메시지, PDF 파일 경로) 튜플
    """
    try:
        pdf_path = session_manager.export_to_pdf()
        if pdf_path:
            return f"PDF가 생성되었습니다: {pdf_path}", pdf_path
        else:
            return "PDF 생성 실패 (필요한 라이브러리가 설치되지 않았을 수 있습니다)", None
    except Exception as e:
        return f"PDF 생성 중 오류: {str(e)}", None


def update_api_key_inputs(vlm_type: str) -> Tuple[gr.update, gr.update, gr.update]:
    """
    VLM 타입에 따라 API 키 입력 필드 표시/숨김
    
    Args:
        vlm_type: 선택된 VLM 타입
    
    Returns:
        각 입력 필드의 업데이트 정보
    """
    if vlm_type == "openai":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif vlm_type == "gemini":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    elif vlm_type == "local":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def create_interface():
    """Gradio 인터페이스 생성"""
    # theme 파라미터는 일부 Gradio 버전에서 지원되지 않으므로 제거
    with gr.Blocks(title="VLM 이미지 캡션 생성기") as app:
        # 키보드 단축키 설정 (Ctrl+Enter 또는 Enter로 캡션 생성)
        app.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            function() {
                document.addEventListener('keydown', function(e) {
                    // Ctrl+Enter 또는 Enter 키 감지
                    if ((e.ctrlKey && e.key === 'Enter') || (e.key === 'Enter' && !e.ctrlKey && !e.shiftKey)) {
                        // 텍스트 입력 필드에 포커스가 있지 않을 때만 실행
                        const activeElement = document.activeElement;
                        const isTextInput = activeElement && (
                            activeElement.tagName === 'TEXTAREA' || 
                            activeElement.tagName === 'INPUT' ||
                            activeElement.isContentEditable
                        );
                        
                        // 텍스트 입력 필드가 아니거나, Ctrl+Enter인 경우에만 실행
                        if (!isTextInput || (e.ctrlKey && e.key === 'Enter')) {
                            e.preventDefault();
                            const generateBtn = document.querySelector('#generate_btn button, button[variant="primary"]');
                            if (generateBtn && !generateBtn.disabled) {
                                generateBtn.click();
                            }
                        }
                    }
                });
            }
            """
        )
        
        gr.Markdown(
            """
            # VLM 이미지 캡션 생성기
            
            이미지(0~5개)와 프롬프트를 입력하여 VLM을 사용해 캡션을 생성합니다.
            
            **단축키**: `Ctrl+Enter` 또는 `Enter` (입력 필드 외부에서)로 캡션 생성
            """
        )
        
        # 세션 관리 섹션
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 프로젝트 관리")
                session_status = gr.Textbox(
                    label="세션 상태",
                    value="세션 없음",
                    interactive=False
                )
                with gr.Row():
                    start_btn = gr.Button("프로젝트 시작", variant="primary")
                    save_btn = gr.Button("프로젝트 저장", variant="secondary")
                    end_btn = gr.Button("프로젝트 종료", variant="stop")
                    export_pdf_btn = gr.Button("PDF 내보내기", variant="secondary")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 이미지 업로드
                image_gallery = gr.Gallery(
                    label="이미지 업로드 (최대 5개, 선택사항)",
                    type="pil",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
                
                # VLM 선택
                vlm_type = gr.Dropdown(
                    choices=["openai", "gemini", "local"],
                    value="openai",
                    label="VLM 선택",
                    info="사용할 Vision Language Model을 선택하세요"
                )
                
                # 시스템 프롬프트 입력 (고정, 계속 사용)
                system_prompt = gr.Textbox(
                    label="시스템 프롬프트 (고정)",
                    placeholder="예: 당신은 이미지 분석 전문가입니다. 이미지를 자세히 분석하여 설명해주세요.",
                    lines=3,
                    value="이 이미지에 대해 자세히 설명해주세요.",
                    info="계속 사용할 기본 프롬프트를 입력하세요. 세션 동안 유지됩니다."
                )
                
                # 질문/컨텍스트 입력 (계속 유지됨)
                context = gr.Textbox(
                    label="질문/컨텍스트 (유지)",
                    placeholder="예: 이 이미지에서 무엇이 보이나요?",
                    lines=2,
                    value="",
                    info="질문이나 컨텍스트를 입력하세요. 이 내용은 계속 유지됩니다."
                )
                
                # 이번 iteration에만 사용할 추가 컨텍스트 (초기화됨)
                iteration_context = gr.Textbox(
                    label="추가 컨텍스트 (이번 iteration만, 생성된 캡션이 자동 복사됨)",
                    placeholder="예: 특히 색상에 집중해서 설명해주세요.",
                    lines=2,
                    value="",
                    info="이번 iteration에만 사용할 추가 컨텍스트를 입력하세요. 생성된 캡션이 자동으로 여기에 복사됩니다."
                )
                
                # API 키 입력 (조건부 표시)
                gr.Markdown("### API 키 설정 (선택사항)")
                gr.Markdown("*환경 변수에 설정되어 있으면 생략 가능합니다.*")
                
                openai_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    visible=True,  # 초기값: openai가 기본값이므로 True
                    info="OpenAI VLM 사용 시 필요 (환경 변수 OPENAI_API_KEY에 설정되어 있으면 생략 가능)"
                )
                gemini_key_input = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    visible=False,
                    info="Gemini VLM 사용 시 필요 (환경 변수 GEMINI_API_KEY에 설정되어 있으면 생략 가능)"
                )
                local_model_input = gr.Textbox(
                    label="로컬 모델 이름",
                    placeholder="Qwen/Qwen-VL-Chat",
                    visible=False,
                    value=Settings.get_local_model(),
                    info="로컬 VLM 모델 이름 (예: Qwen/Qwen-VL-Chat)"
                )
                
                # VLM 타입 변경 시 API 키 입력 필드 업데이트
                vlm_type.change(
                    fn=update_api_key_inputs,
                    inputs=[vlm_type],
                    outputs=[openai_key_input, gemini_key_input, local_model_input]
                )
                
                # 생성 버튼
                generate_btn = gr.Button("캡션 생성 (Ctrl+Enter)", variant="primary", size="lg", elem_id="generate_btn")
            
            with gr.Column(scale=1):
                # 결과 표시
                output = gr.Textbox(
                    label="생성된 캡션",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                # 상태 메시지 표시
                status_msg = gr.Textbox(
                    label="상태 메시지",
                    lines=2,
                    interactive=False,
                    visible=True
                )
        
        # 이벤트 연결
        generate_btn.click(
            fn=create_caption,
            inputs=[
                image_gallery,
                vlm_type,
                system_prompt,
                context,
                iteration_context,
                openai_key_input,
                gemini_key_input,
                local_model_input
            ],
            outputs=[output, context, image_gallery, iteration_context, status_msg]
        )
        
        # 프로젝트 시작 버튼
        start_btn.click(
            fn=start_project,
            inputs=[
                system_prompt,
                context,
                vlm_type,
                openai_key_input,
                gemini_key_input,
                local_model_input
            ],
            outputs=[status_msg, session_status]
        )
        
        # 프로젝트 저장 버튼
        save_btn.click(
            fn=save_project,
            inputs=[],
            outputs=[status_msg, session_status]
        )
        
        # 프로젝트 종료 버튼
        end_btn.click(
            fn=end_project,
            inputs=[],
            outputs=[status_msg, session_status]
        )
        
        # PDF 내보내기 버튼
        export_pdf_btn.click(
            fn=export_pdf,
            inputs=[],
            outputs=[status_msg]
        )
        
        # 예시
        gr.Markdown(
            """
            ## 사용 방법
            
            ### 1. 프로젝트 시작
            1. **시스템 프롬프트**를 입력하세요 (고정, 계속 사용할 기본 지시사항)
            2. **질문/컨텍스트**를 입력하세요 (계속 유지될 질문이나 컨텍스트)
            3. 사용할 VLM을 선택하세요
            4. 필요시 API 키를 입력하세요 (환경 변수에 설정되어 있으면 생략 가능)
            5. **"프로젝트 시작"** 버튼을 클릭하세요
               - 시스템 프롬프트와 질문/컨텍스트를 기반으로 폴더 이름이 생성됩니다
            
            ### 2. 캡션 생성
            1. 이미지를 업로드하세요 (0~5개, 선택사항)
            2. **추가 컨텍스트**를 입력하세요 (이번 iteration에만 사용, 선택사항)
            3. **"캡션 생성"** 버튼을 클릭하세요
            4. 생성된 캡션은 자동으로 클립보드에 복사됩니다
            5. 생성된 캡션과 이미지는 자동으로 기록됩니다 (Iteration별로)
            
            ### 3. 프로젝트 관리
            - **프로젝트 저장**: 현재 세션을 저장합니다
            - **프로젝트 종료**: 세션을 종료합니다
            - **PDF 내보내기**: 현재 세션을 PDF로 내보냅니다
            
            **참고**: 
            - 시스템 프롬프트 + 질문/컨텍스트(유지) + 추가 컨텍스트(초기화)가 자동으로 결합되어 최종 프롬프트로 사용됩니다.
            - 질문/컨텍스트는 계속 유지되며, 추가 컨텍스트만 매번 초기화됩니다.
            - 생성된 캡션은 자동으로 클립보드에 복사됩니다.
            - 모든 기록은 `export_docs/` 폴더에 저장됩니다.
            - 각 iteration은 별도의 섹션으로 기록됩니다.
            """
        )
    
    return app


def main():
    """메인 함수"""
    app = create_interface()
    app.launch(share=True, server_name="0.0.0.0", server_port=3001)


if __name__ == "__main__":
    main()

