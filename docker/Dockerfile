# llama.cpp (master, which includes DFlash 2 since PR #27342 merged) for
# CUDA sm_86 (RTX 3090), built with CUDA 13.4.2 via llama.cpp's CMake build.
# Runtime image ships llama-server, nginx (front door + /monitor/) and the
# hw-metrics sidecar; GGUFs are mounted at /models.

ARG CUDA_VERSION=13.4.2
ARG UBUNTU_VERSION=24.04

# Build stage needs the -devel image: it ships nvcc and the CUDA headers.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# GPU architectures to compile for; semicolon list for mixed cards,
# e.g. "86;89" for an RTX 3090 next to a 4090.
ARG CUDA_ARCH=86
# NCCL for multi-GPU split modes (llama.cpp warns multi-card performance is
# "suboptimal" without it). Single-card runs don't use it. OFF saves ~300 MB.
ARG GGML_CUDA_NCCL=ON
# Branch, tag (e.g. b7000) or commit SHA of ggml-org/llama.cpp to build.
# BuildKit resolves the ref on every build and caches by the commit it points
# to, so a moving ref like `master` rebuilds only when upstream changes.
ARG LLAMA_CPP_REF=master
# Parallel compile jobs. Each nvcc on the MMQ/FA template instances can take
# 2+ GB; -j$(nproc) on 16 threads / 23 GB WSL segfaulted cicc.
ARG BUILD_JOBS=8

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        ccache \
        libssl-dev

ADD --keep-git-dir=true https://github.com/ggml-org/llama.cpp.git#${LLAMA_CPP_REF} /src
WORKDIR /src

# ccache cache mount: rebuilds after a small upstream bump only recompile the
# changed .cu/.cpp files instead of all ~370 objects.
RUN --mount=type=cache,target=/root/.cache/ccache \
    git log -1 --format='llama.cpp %H %s' \
    && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DGGML_CUDA_NCCL="${GGML_CUDA_NCCL}" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_CURL=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined -L/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-shlib-undefined -L/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release --target llama-server -j"${BUILD_JOBS}" \
    && mkdir -p /out/lib \
    && cp -a build/bin/llama-server /out/ \
    && cp -a build/bin/*.so* /out/lib/ \
    && cp -a /usr/local/cuda/lib64/libcudart.so.* \
             /usr/local/cuda/lib64/libcublas.so.* \
             /usr/local/cuda/lib64/libcublasLt.so.* /out/lib/ \
    && if [ "${GGML_CUDA_NCCL}" = "ON" ]; then cp -a /usr/lib/x86_64-linux-gnu/libnccl.so.* /out/lib/; fi

# -base instead of -runtime: llama-server only needs cudart + cuBLAS (+ NCCL),
# which are copied from the build stage above. Skips cuFFT/cuSPARSE/NPP/nvJPEG
# that the -runtime image would add (several GB).
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# Under vLLM, Triton JIT-compiles host stubs (gcc + python3-dev) and FlashInfer
# JIT-compiles CUDA kernels for sm_86 on first use (e.g. its top-p/top-k
# sampler), which needs nvcc + CUDA headers: without them vLLM's engine core
# dies at startup. The -devel image would add ~10 GB; these add ~1 GB.
ARG CUDA_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && cuda_pkg="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -)" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libssl3 \
        curl \
        ca-certificates \
        nginx \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
        "cuda-nvcc-${cuda_pkg}" \
        "cuda-cudart-dev-${cuda_pkg}" \
        "libcurand-dev-${cuda_pkg}"
ENV CUDA_HOME=/usr/local/cuda

# vLLM from its prebuilt wheel into /opt/venv. vLLM pins torch; the cu130
# index supplies the CUDA 13.0 build of it (no cu134 torch exists; the 13.4
# driver runs 13.0 wheels). torch brings its own cuBLAS/cuDNN/NCCL wheels.
# FlashInfer's prebuilt kernels (flashinfer-cubin) only ship on its own index.
# Kept above the llama.cpp COPYs so a llama.cpp bump doesn't reinstall ~10 GB.
ARG VLLM_VERSION=0.31.0
ARG FLASHINFER_VERSION=0.7.0.post1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        --extra-index-url "${TORCH_INDEX_URL}" \
        --extra-index-url https://flashinfer.ai/whl/ \
        "vllm==${VLLM_VERSION}" \
        "flashinfer-cubin==${FLASHINFER_VERSION}" \
    && /opt/venv/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
ENV PATH=/opt/venv/bin:${PATH}

# llama.cpp's own CUDA 13.4 cudart/cuBLAS live in /opt/llama/lib and are found
# through ldconfig, not LD_LIBRARY_PATH: torch resolves its bundled CUDA 13.0
# libraries via RUNPATH first, so the two stacks don't shadow each other.
COPY --from=build /out/llama-server /usr/local/bin/llama-server
COPY --from=build /out/lib/ /opt/llama/lib/
# Fail the build if anything besides libcuda (the host driver, injected by
# `gpus: all` at runtime) is unresolved — the trimmed -base image makes that easy to miss.
RUN echo /opt/llama/lib > /etc/ld.so.conf.d/llama.conf \
    && ldconfig \
    && missing="$(ldd /usr/local/bin/llama-server /opt/llama/lib/*.so* \
        | grep 'not found' | grep -v 'libcuda\.so' | sort -u)" \
    && if [ -n "$missing" ]; then echo "unresolved libs:\n$missing"; exit 1; fi

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# nginx on :8888 fronts the backend (:8080), serves the Inference Overdrive
# monitor at /monitor/ (+ hw-metrics :8083 at /monitor/hw) and the server
# control page at /server/ (+ manager :8090 at /server/api/).
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY python/Dashboard/hw-metrics.py /opt/hw-metrics/hw-metrics.py
COPY python/Dashboard/monitor.html /opt/monitor/index.html
COPY python/Server/manager.py /opt/server/manager.py
COPY python/Server/index.html /opt/server/www/index.html
RUN nginx -t

# One model folder for both backends: the Hugging Face hub cache *is* /models
# (repos land in /models/models--<owner>--<name>), so /server downloads,
# `vllm serve owner/name` and huggingface-cli share one copy; llama.cpp's own
# downloads (-hf / LLAMA_CACHE) land there too. HF_HOME keeps HF's token/xet
# housekeeping out of the listing. CUDA_DEVICE_ORDER numbers cards in PCI
# order like nvidia-smi, so "GPU 1" is the same card on /server, the monitor
# and in both backends.
# HF_TOKEN_PATH: the token saved from /server (an HF_TOKEN env var still wins).
ENV HF_HOME=/models/.huggingface \
    HF_HUB_CACHE=/models \
    LLAMA_CACHE=/models \
    HF_TOKEN_PATH=/state/hf_token \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

WORKDIR /app
EXPOSE 8888

# Healthy once nginx and the server manager answer (/monitor and /server load);
# a running model isn't required — the container may start with none.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=4 \
    CMD curl -fsS http://127.0.0.1:8888/server/api/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
