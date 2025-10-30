# 빠른 시작 가이드

RunPod에서 민감 식별자 탐지 모델을 5분 만에 실행하는 방법

## 📋 필요한 것

1. **RunPod GPU 인스턴스** (NVIDIA GPU 필수)
2. **모델 파일** (2개)
   - `base_model.gguf` (~7.7GB)
   - `lora.gguf` (~수백 MB)
3. **데이터셋**
   - `dataset.jsonl` (~2-5MB, 122개 항목)

---

## 🚀 3단계로 시작하기

### 1️⃣ 환경 설정 (1-2분)

```bash
# 저장소 클론 또는 파일 업로드
cd /workspace

# 실행 권한 부여
chmod +x setup.sh

# 환경 설정 실행
bash setup.sh
```

**예상 출력:**
```
====================================
RunPod GPU 환경 설정 시작
====================================
[1/7] 시스템 패키지 업데이트...
[2/7] Python 기본 패키지 설치...
[3/7] 기존 llama-cpp-python 제거...
[4/7] CUDA 지원 llama-cpp-python 설치...
[5/7] llama-cpp-python 설치 확인...
✅ llama-cpp-python 설치 성공
[6/7] 작업 디렉토리 생성...
[7/7] GPU 환경 확인...

=== NVIDIA GPU 정보 ===
NVIDIA RTX A6000, 535.129.03, 49140 MiB

✅ 환경 설정 완료!
```

---

### 2️⃣ 파일 업로드 (2-3분)

```bash
# 디렉토리 확인
ls -lh models/
ls -lh dataset.jsonl
```

**필요한 파일 3개:**
1. `models/base_model.gguf` (베이스 모델)
2. `models/lora.gguf` (LoRA 어댑터)
3. `dataset.jsonl` (평가 데이터셋)

**업로드 방법:**
- RunPod 웹 UI: Files → Upload
- SCP: `scp dataset.jsonl user@runpod:~/workspace/`

---

### 3️⃣ 추론 실행 (즉시)

```bash
python run_inference.py
```

**예상 출력:**
```
============================================================
모델 로딩 중...
============================================================
Base model: models/base_model.gguf
LoRA adapter: models/lora.gguf
Context size: 12288
GPU layers: ALL
✅ 모델 로딩 완료!

============================================================
데이터셋 처리 시작
============================================================

📊 필터링 결과:
  원본 데이터셋: 122개
  토큰 초과 스킵: 0개
  필터링 후: 122개

📝 처리 상태:
  필터링 후 항목: 122개
  이미 처리됨: 0개
  처리 필요: 122개
  최대 입력 토큰: ~10500 tokens

Processing: 100%|████████████| 122/122 [45:30<00:00, 22.4s/file, ETA=0초, avg=22.4s/file, ids=234]

============================================================
✅ 처리 완료!
============================================================
⏱️  총 소요 시간: 45분 30초
📊 평균 처리 속도: 22.4초/파일
📁 처리된 파일: 122개
🔍 총 민감 식별자 (중복 포함): 567개
✨ 고유 민감 식별자: 234개

📂 출력 파일:
  • output/per_file_results.jsonl
  • output/all_identifiers.json
  • output/summary.json

✨ 평가 완료!
```

---

## 📊 결과 확인

### 요약 정보
```bash
cat output/summary.json
```

```json
{
  "total_files": 122,
  "processed_files": 122,
  "total_sensitive_identifiers": 567,
  "unique_sensitive_identifiers": 234,
  "processing_time": 2730.5,
  "avg_time_per_file": 22.4
}
```

### 파일별 결과 (첫 3개)
```bash
head -n 3 output/per_file_results.jsonl | python -m json.tool
```

### 전체 민감 식별자 (빈도 Top 10)
```bash
python -m json.tool < output/all_identifiers.json | head -n 20
```

```json
{
  "sensitive_identifiers": {
    "apiKey": 15,
    "savePassword": 12,
    "encryptData": 10,
    "privateKey": 8,
    ...
  }
}
```

---

## 🎮 고급 옵션

### GPU 메모리 절약
```bash
python run_inference.py --gpu_layers 24
```

### 최대 토큰 수 조정
```bash
python run_inference.py --max_input_tokens 12000
```

### 체크포인트 초기화 (처음부터)
```bash
python run_inference.py --reset
```

### 커스텀 경로
```bash
python run_inference.py \
  --dataset my_data.jsonl \
  --base_model models/my_base.gguf \
  --lora models/my_lora.gguf \
  --output my_results
```

---

## 🔍 실시간 모니터링

### GPU 사용률 확인
```bash
watch -n 1 nvidia-smi
```

**정상 작동 시:**
- GPU-Util: 80-100%
- Memory-Usage: 15,000+ MiB

**비정상 (CPU 사용) 시:**
- GPU-Util: 0%
- → `bash setup.sh` 재실행 필요!

### 처리 진행상황
```bash
# 처리된 파일 수 확인
wc -l checkpoint/processed.jsonl
```

---

## ⚠️ 문제 해결

### Q1: "llama-cpp-python 설치 실패"
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Q2: GPU 인식 안 됨
```bash
# CUDA 확인
nvidia-smi
nvcc --version

# setup.sh 재실행
bash setup.sh
```

### Q3: 메모리 부족
```bash
# GPU 레이어 줄이기
python run_inference.py --gpu_layers 16
```

### Q4: 처리가 너무 느림
```bash
# GPU 사용 확인
nvidia-smi

# GPU-Util이 0%면 CPU로 돌아가는 것
# → setup.sh 재실행
bash setup.sh
```

---

## 💡 팁

### 1. 중단 후 재개
Ctrl+C로 중단해도 안전합니다. 다음 실행 시 자동으로 이어서 처리됩니다.

### 2. 결과 다운로드
```bash
# 결과 압축
zip -r results.zip output/

# 로컬로 다운로드
# RunPod UI: Files → results.zip 다운로드
```

### 3. 배치 처리
여러 데이터셋을 순차적으로 처리:
```bash
for dataset in dataset1.jsonl dataset2.jsonl dataset3.jsonl; do
  python run_inference.py --dataset $dataset --output output_${dataset%.jsonl}
done
```

---

## 📖 더 알아보기

- [README.md](README.md) - 전체 문서
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 프로젝트 구조
- [Swingft 프로젝트](https://github.com/your-org/swingft)

---

## ✅ 체크리스트

- [ ] RunPod GPU 인스턴스 준비
- [ ] setup.sh 실행 완료
- [ ] 모델 파일 업로드 (base_model.gguf, lora.gguf)
- [ ] 데이터셋 업로드 (dataset.jsonl)
- [ ] GPU 인식 확인 (nvidia-smi)
- [ ] run_inference.py 실행
- [ ] 결과 확인 (output/)
- [ ] 결과 다운로드

---

**완료 시간: 약 50분 (설정 5분 + 추론 45분)**
