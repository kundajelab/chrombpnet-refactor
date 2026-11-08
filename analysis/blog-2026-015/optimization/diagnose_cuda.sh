#!/bin/bash
# Why torch cannot see the GPU. Run it in the environment that failed:
#
#   ./diagnose_cuda.sh                              # chrombpnet-pytorch, as the benchmark
#   CONDA_ENV=chrombpnet-pytorch-sherlock ./diagnose_cuda.sh
#
# torch.cuda.is_available() returns False for several unrelated reasons and
# reports none of them, so this collects the things that distinguish them.
CONDA_ENV="${CONDA_ENV:-chrombpnet-pytorch}"
# chrombpnet-pytorch (PyTorch 2.14) needs glibc >= 2.28 (engines/envs/glibc_scan.py:
# torch and igraph); some of sherlock's GPU nodes (the L40S ones) run 2.17. Stop
# here on such a node rather than failing at the first import.
GLIBC_FLOOR="${GLIBC_FLOOR:-2.28}"
if [[ "${CONDA_ENV}" != "chrombpnet-pytorch-sherlock" ]]; then
  NODE_GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
  if [[ -n "${NODE_GLIBC}" && "$(printf '%s\n%s\n' "${GLIBC_FLOOR}" "${NODE_GLIBC}" | sort -V | head -1)" != "${GLIBC_FLOOR}" ]]; then
    echo "Error: ${CONDA_ENV} needs glibc >= ${GLIBC_FLOOR}; this node ($(hostname)) has ${NODE_GLIBC}." >&2
    echo "       Resubmit with --exclude=$(hostname), or set CONDA_ENV=chrombpnet-pytorch-sherlock." >&2
    exit 1
  fi
fi
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV}"

echo "host              : $(hostname)"
echo "conda env         : ${CONDA_ENV}"
echo "glibc             : $(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo unknown)"
echo "CUDA_VISIBLE_DEVICES : '${CUDA_VISIBLE_DEVICES-<unset>}'"
echo "LD_LIBRARY_PATH   : ${LD_LIBRARY_PATH:-<unset>}"
echo
echo "--- driver / devices ---"
nvidia-smi -L 2>&1 | sed 's/^/  /'
nvidia-smi --query-gpu=name,driver_version,compute_mode,memory.used --format=csv 2>&1 | sed 's/^/  /'
echo
echo "--- can the CUDA driver library be loaded at all? ---"
python - <<'PYEOF'
import ctypes, glob, os
for name in ("libcuda.so.1", "libcuda.so"):
    try:
        ctypes.CDLL(name); print(f"  {name}: loaded")
    except OSError as exc:
        print(f"  {name}: FAILED -- {exc}")
found = glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + glob.glob("/usr/lib64/libcuda.so*")
print("  on disk:", found or "none found in the usual places")
print("  /dev/nvidia*:", sorted(os.path.basename(p) for p in glob.glob("/dev/nvidia*")) or "none")
PYEOF
echo
echo "--- what torch says, with the real error ---"
python - <<'PYEOF'
import chrombpnet, torch, os
print("  torch            :", torch.__version__, "| built for CUDA", torch.version.cuda)
print("  arch list        :", torch.cuda.get_arch_list())
print("  device_count()   :", torch.cuda.device_count())
print("  is_available()   :", torch.cuda.is_available())
# is_available() swallows the cause; init() raises it.
try:
    torch.cuda.init()
    print("  cuda.init()      : ok")
except Exception as exc:
    print(f"  cuda.init()      : {type(exc).__name__}: {exc}")
try:
    print("  raw device count :", torch._C._cuda_getDeviceCount())
except Exception as exc:
    print(f"  raw device count : {type(exc).__name__}: {exc}")
if torch.cuda.is_available():
    torch.zeros(8, device="cuda").sum().item()
    print("  allocation test  : ok on", torch.cuda.get_device_name(0))
PYEOF
