#!/usr/bin/env python3
"""
run_inference.py

RunPod GPU 환경에서 민감 식별자 탐지 모델 실행
- JSONL 데이터셋 입력 (instruction, input 구조)
- 학습 시 사용한 Alpaca 형식과 동일하게 추론
- 파일별 + 전체 식별자 수집
- 중단 시 재개 기능
- 해시 기반 체크포인트
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
from llama_cpp import Llama


class SensitiveIdentifierDetector:
    """Swift 코드에서 민감한 식별자 탐지"""

    def __init__(self,
                 base_model_path: str,
                 lora_path: str,
                 n_ctx: int = 32768,  # 12288 → 32768 (H200 충분)
                 n_gpu_layers: int = -1,
                 checkpoint_file: str = "checkpoint/processed.jsonl"):
        """
        Args:
            base_model_path: 베이스 모델 경로
            lora_path: LoRA 어댑터 경로
            n_ctx: 컨텍스트 크기 (기본 32768, Phi-3-mini-128k 지원)
            n_gpu_layers: GPU 레이어 수 (-1 = 전체)
            checkpoint_file: 처리 완료된 파일 기록
        """
        self.checkpoint_file = checkpoint_file
        self.processed_hashes = self._load_checkpoint()

        print("=" * 60)
        print("모델 로딩 중...")
        print("=" * 60)
        print(f"Base model: {base_model_path}")
        print(f"LoRA adapter: {lora_path}")
        print(f"Context size: {n_ctx}")
        print(f"GPU layers: {n_gpu_layers if n_gpu_layers != -1 else 'ALL'}")

        try:
            self.model = Llama(
                model_path=base_model_path,
                lora_path=lora_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=4,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
            print("✅ 모델 로딩 완료!\n")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise

    def _compute_hash(self, instruction: str, input_text: str) -> str:
        """instruction + input으로 해시 생성"""
        combined = f"{instruction}::{input_text}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _load_checkpoint(self) -> Dict[str, Dict]:
        """체크포인트 로드

        Returns:
            Dict[hash, result]: 해시를 키로, 결과를 값으로
        """
        if not os.path.exists(self.checkpoint_file):
            return {}

        processed = {}
        with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed[data['hash']] = data['result']
                    except:
                        continue

        if processed:
            print(f"✅ 체크포인트 로드: {len(processed)}개 항목 이미 처리됨")

        return processed

    def _save_checkpoint(self, content_hash: str, result: Dict):
        """체크포인트 저장"""
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

        checkpoint_entry = {
            "hash": content_hash,
            "result": result
        }

        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(checkpoint_entry, ensure_ascii=False) + '\n')

        self.processed_hashes[content_hash] = result

    def _should_process(self, instruction: str, input_text: str) -> tuple:
        """처리 여부 확인

        Returns:
            (should_process: bool, cached_result: Dict or None)
        """
        current_hash = self._compute_hash(instruction, input_text)

        if current_hash in self.processed_hashes:
            return False, self.processed_hashes[current_hash]

        return True, None

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        """학습 시 사용한 Alpaca 형식으로 프롬프트 생성

        형식: ### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n
        """
        inst = instruction.strip()
        inp = input_text.strip()

        if inp:
            prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

        return prompt

    def detect_sensitive_identifiers(self, item: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, List[str]]:
        """단일 항목에서 민감 식별자 탐지

        Args:
            item: {"instruction": "...", "input": "..."}
            metadata: {"source_file": "...", "file_path": "..."}

        Returns:
            {
                "source_file": "...",
                "sensitive_identifiers": [...],
                "reasoning": {...}  # 선택사항
            }
        """
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')

        # 체크포인트 확인
        should_process, cached_result = self._should_process(instruction, input_text)

        if not should_process:
            print(f"  💾 캐시된 결과 사용")
            return cached_result

        # 프롬프트 생성 (학습 형식과 동일)
        prompt = self._format_prompt(instruction, input_text)

        try:
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=20480,  # 16384 → 20480 (n_ctx=32768이면 충분)
                temperature=0.1,
                top_p=0.95,
                stop=["<|endoftext|>", "###"],
                echo=False
            )

            output = response['choices'][0]['text'].strip()

            # 출력이 잘렸는지 확인
            finish_reason = response['choices'][0].get('finish_reason', '')
            if finish_reason == 'length':
                print(f"  ⚠️ 출력이 max_tokens에 의해 잘림 (토큰 부족)")

            # JSON 파싱 (sensitive_identifiers + reasoning 추출)
            parsed_output = self._parse_output(output)

            # 결과 딕셔너리 구성
            result = {
                "source_file": metadata.get("source_file", "unknown"),
                "sensitive_identifiers": parsed_output.get("sensitive_identifiers", {}),
                "raw_output": output  # 전체 저장 (확인용)
            }

            # 체크포인트 저장
            content_hash = self._compute_hash(instruction, input_text)
            self._save_checkpoint(content_hash, result)

            return result

        except Exception as e:
            print(f"\n⚠️  에러 발생: {e}")
            return {
                "source_file": metadata.get("source_file", "unknown"),
                "sensitive_identifiers": {},
                "error": str(e)
            }

    def _parse_output(self, output: str) -> Dict:
        """모델 출력에서 sensitive_identifiers 추출

        예상 형식:
        {
          "sensitive_identifiers": {
            "identifier1": {
              "reasoning": "..."
            },
            "identifier2": {
              "reasoning": "..."
            }
          }
        }
        """
        import re

        # JSON 블록 추출 시도
        try:
            # 방법 1: 가장 큰 완전한 JSON 객체 찾기
            json_candidates = []
            depth = 0
            start_idx = -1

            for i, char in enumerate(output):
                if char == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_idx != -1:
                        # 완전한 JSON 객체 발견
                        json_candidates.append(output[start_idx:i + 1])
                        start_idx = -1

            # 가장 긴 JSON 시도 (보통 가장 완전함)
            json_candidates.sort(key=len, reverse=True)

            for json_str in json_candidates:
                try:
                    # 주석 제거
                    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)

                    data = json.loads(json_str)

                    # sensitive_identifiers 키가 있으면 반환
                    if 'sensitive_identifiers' in data:
                        num_ids = len(data['sensitive_identifiers'])
                        if num_ids > 0:
                            print(f"  ✅ {num_ids}개 식별자 파싱 성공")
                        return {"sensitive_identifiers": data['sensitive_identifiers']}

                except json.JSONDecodeError:
                    continue  # 다음 후보 시도

        except Exception as e:
            print(f"  ⚠️ 파싱 에러: {e}")

        # 파싱 실패 시 빈 결과
        print(f"  ⚠️ JSON 파싱 실패 - 출력 미리보기: {output[:200]}...")
        return {"sensitive_identifiers": {}}

    def process_dataset(self,
                        dataset_path: str,
                        output_dir: str = "output",
                        max_input_tokens: int = 10500) -> Dict[str, Any]:
        """전체 데이터셋 처리

        Args:
            dataset_path: 입력 JSONL 파일
            output_dir: 출력 디렉토리
            max_input_tokens: 최대 입력 토큰 수
        """
        print("=" * 60)
        print("데이터셋 처리 시작")
        print("=" * 60)

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 출력 파일 경로
        per_file_output = os.path.join(output_dir, "per_file_results.jsonl")
        all_identifiers_output = os.path.join(output_dir, "all_identifiers.json")
        summary_output = os.path.join(output_dir, "summary.json")

        # 데이터셋 로드 및 필터링
        dataset = []
        skipped_count = 0
        original_count = 0

        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    original_count += 1
                    try:
                        item = json.loads(line)

                        # 토큰 수 대략 추정
                        instruction = item.get('instruction', '')
                        input_text = item.get('input', '')
                        combined = instruction + input_text
                        approx_tokens = len(combined) / 4

                        # 최대 토큰 수 필터링
                        if approx_tokens <= max_input_tokens:
                            dataset.append(item)
                        else:
                            skipped_count += 1
                            print(f"⚠️  토큰 수 초과로 스킵: ~{int(approx_tokens)} tokens")
                    except json.JSONDecodeError:
                        continue

        print(f"\n📊 필터링 결과:")
        print(f"  원본 데이터셋: {original_count}개")
        print(f"  토큰 초과 스킵: {skipped_count}개")
        print(f"  필터링 후: {len(dataset)}개\n")

        total_items = len(dataset)

        # 처리 필요한 항목 계산
        need_processing = []
        for item in dataset:
            should_process, _ = self._should_process(
                item.get('instruction', ''),
                item.get('input', '')
            )
            if should_process:
                need_processing.append(item)

        already_processed = total_items - len(need_processing)

        print(f"📝 처리 상태:")
        print(f"  필터링 후 항목: {total_items}개")
        print(f"  이미 처리됨: {already_processed}개")
        print(f"  처리 필요: {len(need_processing)}개")
        print(f"  최대 입력 토큰: ~{max_input_tokens} tokens\n")

        # 파일별 결과 수집
        per_file_results = []
        all_sensitive_identifiers = {}  # {identifier: count}

        if len(need_processing) == 0:
            print("✅ 모든 항목이 이미 처리되었습니다!")
            return self._load_existing_results(output_dir)

        # 추론 실행
        import time
        start_time = time.time()
        processed_count = 0

        with tqdm(total=len(need_processing), desc="Processing",
                  unit="file", ncols=100) as pbar:
            for item in need_processing:
                item_start = time.time()

                # metadata 추출
                metadata = item.get('metadata', {})

                # 민감 식별자 탐지
                result = self.detect_sensitive_identifiers(item, metadata)
                per_file_results.append(result)

                # 전체 식별자 수집
                sensitive_ids = result.get("sensitive_identifiers", {})
                for identifier in sensitive_ids.keys():
                    all_sensitive_identifiers[identifier] = all_sensitive_identifiers.get(identifier, 0) + 1

                processed_count += 1

                # 예상 완료 시간 계산
                elapsed = time.time() - start_time
                if processed_count > 0:
                    remaining_items = len(need_processing) - processed_count
                    avg_time_per_item = elapsed / processed_count
                    eta_seconds = remaining_items * avg_time_per_item
                    eta_str = self._format_time(eta_seconds)
                    pbar.set_postfix({
                        'ETA': eta_str,
                        'avg': f'{avg_time_per_item:.1f}s/file',
                        'ids': len(all_sensitive_identifiers)
                    })

                pbar.update(1)

                # 실시간 저장 (10개마다)
                if processed_count % 10 == 0:
                    self._save_results(per_file_results, all_sensitive_identifiers,
                                       per_file_output, all_identifiers_output)

        # 최종 저장
        self._save_results(per_file_results, all_sensitive_identifiers,
                           per_file_output, all_identifiers_output)

        total_time = time.time() - start_time
        time_str = self._format_time(total_time)

        # 통계 계산
        total_identifiers_count = sum(
            len(r.get("sensitive_identifiers", {}))
            for r in per_file_results
        )

        summary = {
            "total_files": total_items,
            "processed_files": len(per_file_results),
            "total_sensitive_identifiers": total_identifiers_count,
            "unique_sensitive_identifiers": len(all_sensitive_identifiers),
            "processing_time": total_time,
            "avg_time_per_file": total_time / total_items if total_items > 0 else 0,
            "output_files": {
                "per_file": per_file_output,
                "all_identifiers": all_identifiers_output,
                "summary": summary_output
            }
        }

        # 요약 저장
        with open(summary_output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print("✅ 처리 완료!")
        print("=" * 60)
        print(f"⏱️  총 소요 시간: {time_str}")
        print(f"📊 평균 처리 속도: {summary['avg_time_per_file']:.1f}초/파일")
        print(f"📁 처리된 파일: {summary['processed_files']}개")
        print(f"🔍 총 민감 식별자 (중복 포함): {summary['total_sensitive_identifiers']}개")
        print(f"✨ 고유 민감 식별자: {summary['unique_sensitive_identifiers']}개")
        print(f"\n📂 출력 파일:")
        print(f"  • {per_file_output}")
        print(f"  • {all_identifiers_output}")
        print(f"  • {summary_output}")

        return summary

    def _save_results(self, per_file_results: List[Dict],
                      all_identifiers: Dict[str, int],
                      per_file_output: str,
                      all_identifiers_output: str):
        """결과 저장"""
        # 파일별 결과
        with open(per_file_output, 'w', encoding='utf-8') as f:
            for result in per_file_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # 전체 식별자 (빈도순 정렬)
        sorted_identifiers = dict(
            sorted(all_identifiers.items(), key=lambda x: x[1], reverse=True)
        )

        with open(all_identifiers_output, 'w', encoding='utf-8') as f:
            json.dump({
                "sensitive_identifiers": sorted_identifiers,
                "total_unique": len(sorted_identifiers),
                "total_count": sum(sorted_identifiers.values())
            }, f, ensure_ascii=False, indent=2)

    def _format_time(self, seconds: float) -> str:
        """초를 읽기 쉬운 형식으로 변환"""
        if seconds < 60:
            return f"{int(seconds)}초"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}분 {secs}초"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}시간 {minutes}분"

    def _load_existing_results(self, output_dir: str) -> Dict[str, Any]:
        """기존 결과 로드"""
        summary_file = os.path.join(output_dir, "summary.json")

        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        return {}


def main():
    parser = argparse.ArgumentParser(description='민감 식별자 탐지 모델 평가 (GPU)')

    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset.jsonl',
        help='입력 JSONL 파일 (기본값: dataset.jsonl)'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='models/base_model.gguf',
        help='베이스 모델 경로'
    )
    parser.add_argument(
        '--lora',
        type=str,
        default='models/lora.gguf',
        help='LoRA 어댑터 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--ctx',
        type=int,
        default=32768,  # 12288 → 32768
        help='컨텍스트 크기 (기본값: 32768, Phi-3-mini-128k 최대 128k 지원)'
    )
    parser.add_argument(
        '--gpu_layers',
        type=int,
        default=-1,
        help='GPU 레이어 수 (기본값: -1 = 전체)'
    )
    parser.add_argument(
        '--max_input_tokens',
        type=int,
        default=10500,
        help='최대 입력 토큰 수 (기본값: 10500)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='체크포인트 초기화'
    )

    args = parser.parse_args()

    # 체크포인트 초기화
    if args.reset:
        checkpoint_file = "checkpoint/processed.jsonl"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("✅ 체크포인트 초기화됨\n")

    # 파일 존재 확인
    if not os.path.exists(args.dataset):
        print(f"❌ 데이터셋을 찾을 수 없습니다: {args.dataset}")
        return

    if not os.path.exists(args.base_model):
        print(f"❌ 베이스 모델을 찾을 수 없습니다: {args.base_model}")
        return

    if not os.path.exists(args.lora):
        print(f"❌ LoRA 어댑터를 찾을 수 없습니다: {args.lora}")
        return

    try:
        # 탐지기 초기화 및 실행
        detector = SensitiveIdentifierDetector(
            base_model_path=args.base_model,
            lora_path=args.lora,
            n_ctx=args.ctx,
            n_gpu_layers=args.gpu_layers
        )

        results = detector.process_dataset(
            dataset_path=args.dataset,
            output_dir=args.output,
            max_input_tokens=args.max_input_tokens
        )

        print(f"\n✨ 평가 완료!")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다")
        print("💾 진행상황이 저장되었습니다. 다음에 다시 실행하면 이어서 처리됩니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()