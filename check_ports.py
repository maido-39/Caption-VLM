#!/usr/bin/env python3
"""
포트 바인딩 가능 여부 확인 스크립트
교내 IDC 서버에서 열 수 있는(바인딩 가능한) 포트를 확인합니다.
"""

import socket
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import time
import errno


def check_port_bindable(host: str, port: int) -> Tuple[int, bool, str]:
    """
    단일 포트의 바인딩 가능 여부를 확인합니다.
    
    Args:
        host: 바인딩할 호스트 주소 (보통 '0.0.0.0' 또는 'localhost')
        port: 포트 번호
    
    Returns:
        (port, is_bindable, error_message) 튜플
    """
    try:
        # 소켓 생성 및 바인딩 시도
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind((host, port))
            sock.close()
            return (port, True, "")
        except OSError as e:
            sock.close()
            if e.errno == errno.EADDRINUSE:
                return (port, False, "이미 사용 중")
            elif e.errno == errno.EACCES:
                return (port, False, "권한 없음 (1024 미만 포트는 root 권한 필요)")
            else:
                return (port, False, f"오류: {e}")
    except Exception as e:
        return (port, False, f"예외 발생: {e}")


def scan_ports(host: str, ports: List[int], max_workers: int = 50) -> List[Tuple[int, bool, str]]:
    """
    여러 포트의 바인딩 가능 여부를 병렬로 확인합니다.
    
    Args:
        host: 바인딩할 호스트 주소
        ports: 확인할 포트 목록
        max_workers: 동시 스캔 스레드 수
    
    Returns:
        (port, is_bindable, error_message) 튜플 리스트
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_port = {
            executor.submit(check_port_bindable, host, port): port 
            for port in ports
        }
        
        completed = 0
        total = len(ports)
        
        for future in as_completed(future_to_port):
            result = future.result()
            results.append(result)
            completed += 1
            port = result[0]
            is_bindable = result[1]
            status = "✓ 사용 가능" if is_bindable else "✗ 사용 불가"
            print(f"[{status}] 포트 {port:5d} ({completed}/{total})", end="\r")
    
    return sorted(results, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(
        description="포트 바인딩 가능 여부 확인 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 특정 포트 범위 확인 (기본: 0.0.0.0에 바인딩)
  python check_ports.py --range 8000 9000
  
  # localhost에만 바인딩 가능한 포트 확인
  python check_ports.py --range 8000 9000 --host localhost
  
  # 특정 포트 목록 확인
  python check_ports.py --ports 8000 8080 9000 7860
  
  # 전체 포트 범위 확인 (주의: 시간이 오래 걸릴 수 있음)
  python check_ports.py --range 1024 65535
  
  # 동시 스캔 스레드 수 조정
  python check_ports.py --range 8000 9000 --workers 100
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="바인딩할 호스트 주소 (기본값: 0.0.0.0, 모든 인터페이스)"
    )
    
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        help="확인할 포트 목록 (예: --ports 8000 8080 9000)"
    )
    
    parser.add_argument(
        "--range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="확인할 포트 범위 (예: --range 8000 9000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="동시 스캔 스레드 수 (기본값: 100)"
    )
    
    parser.add_argument(
        "--show-unavailable",
        action="store_true",
        help="사용 불가능한 포트도 상세히 표시"
    )
    
    args = parser.parse_args()
    
    # 확인할 포트 목록 구성
    ports_to_scan = set()
    
    if args.ports:
        ports_to_scan.update(args.ports)
    
    if args.range:
        start, end = args.range
        if start < 0 or end > 65535:
            print(f"오류: 포트 번호는 0-65535 범위여야 합니다.")
            sys.exit(1)
        if start > end:
            print(f"오류: 시작 포트({start})가 끝 포트({end})보다 큽니다.")
            sys.exit(1)
        ports_to_scan.update(range(start, end + 1))
    
    if not ports_to_scan:
        print("오류: 확인할 포트가 지정되지 않았습니다.")
        print("--ports 또는 --range 옵션으로 포트를 지정하세요.")
        print("\n예시:")
        print("  python check_ports.py --range 8000 9000")
        print("  python check_ports.py --ports 8000 8080 9000")
        sys.exit(1)
    
    ports_to_scan = sorted(list(ports_to_scan))
    
    print(f"\n{'='*70}")
    print(f"포트 바인딩 가능 여부 확인")
    print(f"호스트: {args.host}")
    print(f"확인할 포트 수: {len(ports_to_scan)}개")
    print(f"포트 범위: {min(ports_to_scan)} ~ {max(ports_to_scan)}")
    print(f"동시 스레드 수: {args.workers}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    results = scan_ports(args.host, ports_to_scan, args.workers)
    elapsed_time = time.time() - start_time
    
    # 결과 출력
    print("\n" + "="*70)
    print("스캔 결과")
    print("="*70)
    
    available_ports = []
    unavailable_ports = []
    
    for port, is_bindable, error_msg in results:
        if is_bindable:
            available_ports.append(port)
            print(f"✓ 포트 {port:5d} - 사용 가능 (바인딩 가능)")
        else:
            unavailable_ports.append((port, error_msg))
            if args.show_unavailable:
                print(f"✗ 포트 {port:5d} - 사용 불가 ({error_msg})")
    
    print("\n" + "="*70)
    print("요약")
    print("="*70)
    print(f"사용 가능한 포트: {len(available_ports)}개")
    print(f"사용 불가능한 포트: {len(unavailable_ports)}개")
    
    if available_ports:
        print(f"\n사용 가능한 포트 목록:")
        # 연속된 포트 범위로 그룹화하여 출력
        ranges = []
        start = available_ports[0]
        end = available_ports[0]
        
        for port in available_ports[1:]:
            if port == end + 1:
                end = port
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = port
                end = port
        
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        # 여러 줄로 출력 (너무 길면)
        if len(ranges) <= 20:
            print("  " + ", ".join(ranges))
        else:
            for i, r in enumerate(ranges):
                if i > 0 and i % 10 == 0:
                    print()
                print(f"  {r}", end=", " if i < len(ranges) - 1 else "\n")
            if len(ranges) % 10 != 0:
                print()
    else:
        print("\n사용 가능한 포트가 없습니다.")
    
    print(f"\n총 소요 시간: {elapsed_time:.2f}초")
    print("="*70)


if __name__ == "__main__":
    main()

