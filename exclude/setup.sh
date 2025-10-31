#!/bin/bash
# setup.sh - RunPod GPU 환경 설정 스크립트 (CUDA 지원)

echo "======================================"
echo "RunPod GPU 환경 설정 시작"
echo "======================================"

# 시스템 업데이트
echo "[1/7] 시스템 패키지 업데이트..."
apt-get update -qq

# Python 필수 패키지 설치
echo "[2/7] Python 기본 패키지 설치..."
pip install --upgrade pip -q
pip install tqdm psutil -q

# 기존 llama-cpp-python 제거
echo "[3/7] 기존 llama-cpp-python 제거..."
pip uninstall llama-cpp-python -y -q 2>/dev/null || true

# CUDA 지원 llama-cpp-python 설치
echo "[4/7] CUDA 지원 llama-cpp-python 설치..."
echo "  (이 단계는 2-5분 소요될 수 있습니다)"
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 설치 확인
echo "[5/7] llama-cpp-python 설치 확인..."
python3 -c "from llama_cpp import Llama; print('✅ llama-cpp-python 설치 성공')" || {
    echo "❌ llama-cpp-python 설치 실패!"
    echo "재시도 중 (소스 빌드)..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
}

# 디렉토리 생성
echo "[6/7] 작업 디렉토리 생성..."
mkdir -p models
mkdir -p output
mkdir -p checkpoint

# GPU 확인
echo "[7/7] GPU 환경 확인..."
echo ""
echo "=== NVIDIA GPU 정보 ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "⚠️  nvidia-smi 실행 실패"

echo ""
echo "=== CUDA 버전 ==="
nvcc --version 2>/dev/null | grep "release" || echo "CUDA: $(cat /usr/local/cuda/version.txt 2>/dev/null || echo 'N/A')"

echo ""
echo "=== PyTorch CUDA 확인 (선택사항) ==="
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not installed (optional)"

echo ""
echo "======================================"
echo "✅ 환경 설정 완료!"
echo "======================================"
echo ""
echo "📋 다음 단계:"
echo "1. models/ 디렉토리에 base_model.gguf와 lora.gguf 업로드"
echo "2. dataset.jsonl 파일 업로드"
echo "3. python run_inference.py 실행"
echo ""
echo "🔍 GPU 실시간 모니터링:"
echo "   watch -n 1 nvidia-smi"
echo ""