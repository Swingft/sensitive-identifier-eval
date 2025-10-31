# 프로젝트 구조

```
sensitive-identifier-detection-eval/
│
├── README.md                       # 프로젝트 설명서
├── .gitignore                      # Git 제외 파일
├── requirements.txt                # Python 패키지 목록
├── setup.sh                        # 환경 설정 스크립트
├── run_inference.py                # 메인 추론 스크립트
│
├── models/                         # 모델 파일 (업로드 필요)
│   ├── base_model.gguf             # Phi-3-mini-128k-instruct (GGUF)
│   └── lora.gguf                   # 학습된 LoRA 어댑터 (GGUF)
│
├── dataset.jsonl                   # 입력 데이터셋 (업로드 필요)
│                                   # 형식: {"instruction": "...", "input": "...", "output": "", "metadata": {...}}
│
├── checkpoint/                     # 진행상황 자동 저장
│   └── processed.jsonl             # 처리 완료 항목 (해시 + 결과)
│                                   # 형식: {"hash": "...", "result": {...}}
│
└── output/                         # 결과 출력 디렉토리
    ├── per_file_results.jsonl      # 파일별 탐지 결과
    │                               # 형식: {"source_file": "...", "sensitive_identifiers": {...}}
    │
    ├── all_identifiers.json        # 전체 민감 식별자 집계 (빈도순)
    │                               # 형식: {"sensitive_identifiers": {"id": count, ...}}
    │
    └── summary.json                # 평가 통계 요약
                                    # 형식: {"total_files": N, "unique_sensitive_identifiers": M, ...}
```

## 파일 설명

### 실행 파일
- **setup.sh**: RunPod 환경 초기 설정 (CUDA llama-cpp-python 설치)
- **run_inference.py**: 민감 식별자 탐지 실행 스크립트

### 입력 파일
- **dataset.jsonl**: 평가 데이터셋 (122개 Swift 파일)
- **models/base_model.gguf**: Phi-3-mini-128k-instruct 베이스 모델
- **models/lora.gguf**: 학습된 LoRA 어댑터

### 출력 파일
- **output/per_file_results.jsonl**: 각 파일별 민감 식별자
- **output/all_identifiers.json**: 전체 민감 식별자 (빈도 통계)
- **output/summary.json**: 평가 요약 (파일 수, 식별자 수, 처리 시간 등)

### 체크포인트
- **checkpoint/processed.jsonl**: 처리 완료 항목 (중단 후 재개용)

## 실행 순서

```bash
# 1. 환경 설정
chmod +x setup.sh
bash setup.sh

# 2. 파일 업로드
# - models/base_model.gguf
# - models/lora.gguf  
# - dataset.jsonl

# 3. 추론 실행
python run_inference.py

# 4. 결과 확인
cat output/summary.json
head output/per_file_results.jsonl
python -m json.tool < output/all_identifiers.json
```

## 디렉토리 역할

### `/models`
- 모델 파일 저장소
- GGUF 형식의 베이스 모델과 LoRA 어댑터

### `/checkpoint`
- 처리 진행상황 자동 저장
- Ctrl+C로 중단해도 다음 실행 시 이어서 처리

### `/output`
- 평가 결과 저장
- 파일별 결과, 전체 집계, 통계 요약 포함

## 데이터 흐름

```
dataset.jsonl
    ↓
run_inference.py (모델 추론)
    ↓
checkpoint/processed.jsonl (중간 저장)
    ↓
output/
    ├── per_file_results.jsonl  (파일별)
    ├── all_identifiers.json    (전체)
    └── summary.json            (요약)
```
