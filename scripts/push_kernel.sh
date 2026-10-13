#!/usr/bin/env bash
# Push the server kernel to Kaggle, handling the KGAT bearer-auth trap.
#
# Usage:
#   KAGGLE_API_TOKEN='KGAT_xxxxxxxx...' bash scripts/push_kernel.sh
# or with the classic username/key pair:
#   KAGGLE_USERNAME='you' KAGGLE_KEY='your-api-key' bash scripts/push_kernel.sh
#
set -euo pipefail

KERNEL_DIR="${KERNEL_DIR:-kernel}"
KERNEL_SLUG="${KERNEL_SLUG:-ajithkumaevnr2001/minicpmv47-server-dual-t4}"

# --- 1. The stock CLI only speaks basic auth; KGAT_ tokens are bearer-only.
#        The bundled kagglesdk client accepts KAGGLE_API_TOKEN natively, so we
#        give the CLI placeholders it can parse and let the SDK do real auth.
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-kaggle}"
export KAGGLE_KEY="${KAGGLE_KEY:-placeholder}"
if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  echo "[push] using KAGGLE_API_TOKEN (bearer auth via kagglesdk)"
else
  echo "[push] KAGGLE_API_TOKEN not set - falling back to KAGGLE_USERNAME/KAGGLE_KEY"
fi

# --- 2. sanity: auth must work before we push
if ! kaggle kernels list --mine >/dev/null 2>&1; then
  echo "[push] auth failed - check your token (KGAT_... requires KAGGLE_API_TOKEN)" >&2
  exit 1
fi
echo "[push] auth OK"

# --- 3. push (a new version is created AND immediately runs)
kaggle kernels push -p "$KERNEL_DIR"

echo "[push] submitted. Track it:"
echo "       kaggle kernels status $KERNEL_SLUG"
echo "       kaggle kernels output $KERNEL_SLUG -p out/"
