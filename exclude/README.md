# Sensitive Identifier Detection Evaluation

RunPod GPU 환경에서 민감 식별자 탐지 모델 평가

## ✨ 주요 특징

- ✅ **학습 형식 일치**: Alpaca 형식 (### Instruction / ### Input / ### Response)
- ✅ **파일별 결과**: 각 Swift 파일마다 탐지된 민감 식별자 저장
- ✅ **전체 집계**: 모든 파일의 민감 식별자를 빈도순으로 정렬
- ✅ **해시 기반 체크포인트**: 동일한 입력은 자동으로 스킵
- 🔄 **안전한 중단/재개**: Ctrl+C로 중단해도 진행상황 보존
- 🚀 **GPU 가속**: CUDA 지원으로 빠른 추론
- 📊 **토큰 수 필터링**: 학습 시 사용한 토큰 제한 준수

## 📁 프로젝트 구조

```
sensitive-identifier-detection-eval/
├── setup.sh                  # 환경 설정 스크립트
├── run_inference.py          # 메인 추론 스크립트
├── requirements.txt          # Python 패키지
├── README.md                 # 이 파일
├── .gitignore                # Git 제외 파일
├── models/                   # 모델 파일 (업로드 필요)
│   ├── base_model.gguf       # Phi-3-mini-128k-instruct (GGUF)
│   └── lora.gguf             # 학습된 LoRA 어댑터 (GGUF)
├── dataset.jsonl             # 입력 데이터셋 (업로드 필요)
├── checkpoint/               # 진행상황 저장
│   └── processed.jsonl       # 처리 완료 항목 (해시 + 결과)
└── output/                   # 결과 출력
    ├── per_file_results.jsonl       # 파일별 결과
    ├── all_identifiers.json         # 전체 식별자 (빈도순)
    └── summary.json                 # 통계 요약
```

## 🚀 빠른 시작

### 1단계: 환경 설정

```bash
chmod +x setup.sh
bash setup.sh
```

### 2단계: 파일 업로드

1. **모델 파일** → `models/` 디렉토리
   - `base_model.gguf` (Phi-3-mini-128k-instruct GGUF 버전)
   - `lora.gguf` (학습된 LoRA 어댑터 GGUF 버전)

2. **데이터셋** → 프로젝트 루트
   - `dataset.jsonl` (instruction, input, metadata 포함)

### 3단계: 추론 실행

```bash
python run_inference.py
```

---

## 📊 입력 데이터 형식

### JSONL 형식 (Alpaca)
```jsonl
{"instruction": "Analyze the Swift code and identify all security-sensitive identifiers...", "input": "**Swift Source Code:**\n```swift\n...\n```\n\n**AST Symbol Information:**\n...", "output": "", "metadata": {"source_file": "Auth_Weak_1.swift", "file_path": "/path/to/code/Auth_Weak_1.swift"}}
```

**필수 필드:**
- `instruction`: 모델에게 주는 지시사항
- `input`: Swift 코드 + AST Symbol Information
- `output`: 빈 문자열 (추론 시에는 무시됨)
- `metadata`: 파일 추적 정보 (source_file, file_path)

**프롬프트 형식 (자동 생성):**
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

---

## 📤 출력 형식

### 1. per_file_results.jsonl
파일별 탐지 결과 (JSONL 형식, 한 줄에 하나의 파일)

```jsonl
{"source_file": "Auth_Weak_1.swift", "sensitive_identifiers": {"savePassword": {"reasoning": "..."}, "apiKey": {"reasoning": "..."}}, "raw_output": "..."}
{"source_file": "Crypto_Weak_2.swift", "sensitive_identifiers": {"encryptData": {"reasoning": "..."}, "privateKey": {"reasoning": "..."}}, "raw_output": "..."}
```

**구조:**
- `source_file`: 원본 Swift 파일명
- `sensitive_identifiers`: 민감 식별자와 reasoning
- `raw_output`: 모델의 원본 출력 (디버깅용, 최대 500자)

### 2. all_identifiers.json
전체 민감 식별자 집계 (빈도순 정렬)

```json
{
  "sensitive_identifiers": {
    "apiKey": 15,
    "savePassword": 12,
    "encryptData": 10,
    "privateKey": 8,
    ...
  },
  "total_unique": 234,
  "total_count": 567
}
```

**구조:**
- `sensitive_identifiers`: {식별자: 출현 횟수} (빈도 내림차순)
- `total_unique`: 고유 민감 식별자 수
- `total_count`: 총 민감 식별자 수 (중복 포함)

### 3. summary.json
평가 통계 요약

```json
{
  "total_files": 122,
  "processed_files": 122,
  "total_sensitive_identifiers": 567,
  "unique_sensitive_identifiers": 234,
  "processing_time": 3456.78,
  "avg_time_per_file": 28.3,
  "output_files": {
    "per_file": "output/per_file_results.jsonl",
    "all_identifiers": "output/all_identifiers.json",
    "summary": "output/summary.json"
  }
}
```

---

## 🎛️ 사용법

### 기본 실행
```bash
python run_inference.py
```

### 커스텀 옵션
```bash
python run_inference.py \
  --dataset my_dataset.jsonl \
  --base_model models/phi-3-mini-128k-instruct.gguf \
  --lora models/phi3_lora_adapter.gguf \
  --output my_output \
  --ctx 12288 \
  --max_input_tokens 10500 \
  --gpu_layers -1
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dataset` | 입력 JSONL 파일 | `dataset.jsonl` |
| `--base_model` | 베이스 모델 경로 | `models/base_model.gguf` |
| `--lora` | LoRA 어댑터 경로 | `models/lora.gguf` |
| `--output` | 출력 디렉토리 | `output` |
| `--ctx` | 컨텍스트 크기 | `12288` |
| `--max_input_tokens` | 최대 입력 토큰 | `10500` |
| `--gpu_layers` | GPU 레이어 수 | `-1` (전체) |
| `--reset` | 체크포인트 초기화 | `False` |

---

## 🔐 해시 기반 체크포인트

### 작동 원리

1. **해시 생성**: `instruction + input`을 SHA-256 해싱
2. **자동 스킵**: 동일한 해시는 재처리하지 않음
3. **결과 캐싱**: 체크포인트에 해시 + 결과 저장

### 체크포인트 파일 형식

```jsonl
{"hash": "a1b2c3...", "result": {"source_file": "...", "sensitive_identifiers": {...}}}
{"hash": "d4e5f6...", "result": {"source_file": "...", "sensitive_identifiers": {...}}}
```

---

## 💾 중단 및 재개

### 자동 체크포인트
- 처리 완료된 항목은 `checkpoint/processed.jsonl`에 자동 저장
- **Ctrl+C로 중단해도 안전**
- 다음 실행 시 자동으로 이어서 처리

### 체크포인트 초기화
```bash
python run_inference.py --reset
```

---

## 📈 토큰 수 관리

### 학습 시 설정
- **max_length**: 12288 tokens
- **실제 데이터**: ~10500 tokens 이하로 필터링됨

### 추론 시 설정
- **n_ctx**: 12288 (입력 컨텍스트)
- **max_tokens**: 8192 (출력 생성)
- **입력 필터링**: ~10500 tokens 이하만 처리

---

## 🔧 GPU 설정

### NVIDIA GPU 확인
```bash
nvidia-smi
```

### GPU 메모리 최적화
```bash
# 모든 레이어를 GPU에 로드 (권장)
--gpu_layers -1

# 일부 레이어만 GPU에 로드
--gpu_layers 32
```

---

## ⚠️ 문제 해결

### CUDA 에러
```bash
# CUDA 버전 확인
nvidia-smi

# llama-cpp-python 재설치 (CUDA 지원)
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### 메모리 부족
```bash
# GPU 레이어 수 줄이기
python run_inference.py --gpu_layers 24

# 또는 컨텍스트 크기 줄이기 (비권장)
python run_inference.py --ctx 8192
```

---

## 📝 예시 워크플로우

```bash
# 1. 환경 설정
bash setup.sh

# 2. 파일 업로드 확인
ls models/
ls dataset.jsonl

# 3. 데이터셋 확인
head -n 1 dataset.jsonl | python -m json.tool

# 4. 추론 실행
python run_inference.py

# 5. 결과 확인
cat output/summary.json
head -n 10 output/per_file_results.jsonl
python -m json.tool < output/all_identifiers.json | head -n 50

# 6. 결과 다운로드
# output/ 디렉토리를 로컬로 다운로드
```

---

## 🎯 학습 형식과의 일치

### 학습 시 사용한 `format_example` 함수
```python
def format_example(ex):
    inst = ex.get("instruction")
    inp = ex.get("input")
    out = ex.get("output")
    
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}<|endoftext|>"
    else:
        return f"### Instruction:\n{inst}\n\n### Response:\n{out}<|endoftext|>"
```

### 추론 시 사용하는 `_format_prompt` 메서드
```python
def _format_prompt(self, instruction: str, input_text: str) -> str:
    inst = instruction.strip()
    inp = input_text.strip()
    
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{inst}\n\n### Response:\n"
```

✅ 완전히 동일한 형식!

---

## 💡 평가 데이터셋 정보

- **총 파일**: 122개
- **카테고리 커버리지**: 100% (29/29 MASVS 카테고리)
- **소스**:
  - OWASP 프로젝트: 89개 (DVIA-v2, iGoat-Swift, iBugBazaar)
  - GitHub 프로덕션: 33개 (Signal-iOS, WordPress-iOS 등)

---

## 🤝 관련 프로젝트

- [Swingft](https://github.com/your-org/swingft) - Swift 코드 난독화 프로젝트

---

## 📄 라이센스

MIT
