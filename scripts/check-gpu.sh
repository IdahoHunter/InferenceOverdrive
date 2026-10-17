#!/usr/bin/env bash
# Report whether this host can actually run the RTX CUDA backend.
set -u

echo "== GPU (kernel) =="
if [[ -r /proc/driver/nvidia/gpus ]]; then
  for info in /proc/driver/nvidia/gpus/*/information; do
    [[ -r "$info" ]] || continue
    grep -E '^(Model|Bus Location|Device Minor):' "$info"
  done
else
  echo "no NVIDIA kernel GPU nodes under /proc/driver/nvidia/gpus"
fi

echo
echo "== Driver versions =="
kernel="$(awk '/NVRM version/ {print $NF, $(NF-1), $(NF-2), $(NF-3), $(NF-4)}' /proc/driver/nvidia/version 2>/dev/null || true)"
if [[ -r /proc/driver/nvidia/version ]]; then
  grep -E '^NVRM version' /proc/driver/nvidia/version
fi
user="$(dpkg-query -W -f='${Version}' libnvidia-compute 2>/dev/null || true)"
if [[ -n "$user" ]]; then
  echo "userspace libnvidia-compute: $user"
fi

echo
echo "== nvidia-smi =="
if nvidia-smi >/tmp/inference-overdrive-nvidia-smi.out 2>/tmp/inference-overdrive-nvidia-smi.err; then
  head -20 /tmp/inference-overdrive-nvidia-smi.out
  echo "nvidia-smi: ok"
else
  cat /tmp/inference-overdrive-nvidia-smi.err
  echo "nvidia-smi: FAILED"
fi

echo
echo "== Docker GPU =="
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/tmp/inference-overdrive-docker-gpu.out 2>/tmp/inference-overdrive-docker-gpu.err; then
  head -20 /tmp/inference-overdrive-docker-gpu.out
  echo "docker --gpus all: ok"
else
  tail -20 /tmp/inference-overdrive-docker-gpu.err
  echo "docker --gpus all: FAILED"
fi

echo
echo "== Verdict =="
if grep -qi 'Driver/library version mismatch' /tmp/inference-overdrive-nvidia-smi.err /tmp/inference-overdrive-docker-gpu.err 2>/dev/null; then
  echo "NVIDIA userspace and kernel module do not match."
  echo "nvidia-smi and docker --gpus all fail, but CUDA can still work if the"
  echo "container gets /dev/nvidia* plus host libcuda.so.1 (see docker-compose.yml)."
  echo "A reboot would also align the kernel module with userspace 610."
  exit 0
fi

if ! grep -q 'nvidia-smi: ok' <<<"$(cat /tmp/inference-overdrive-nvidia-smi.out 2>/dev/null; echo)"; then
  if [[ ! -s /tmp/inference-overdrive-nvidia-smi.out ]]; then
    echo "GPU is not usable yet. See the errors above."
    exit 2
  fi
fi

echo "GPU looks usable."
exit 0
