# =====================================================================
#  MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-GGUF
#  Long-running dual-GPU (Kaggle T4 x2) llama.cpp server
#
#  MODES (env var MODE):
#    server (default) - install, download, start llama-server on BOTH T4s,
#                       self-test, then keep the server alive until Kaggle
#                       ends the session (12h max).
#    test            - install, download, benchmark + functional tests, exit.
#
#  NO SECRETS: the HF repo is public, so no tokens are needed.
#  Everything is downloaded to /tmp (scratch; not persisted between sessions).
# =====================================================================
import os, re, sys, time, json, glob, subprocess, signal
from datetime import datetime, timezone

MODE      = os.environ.get("MODE", "server").lower()   # server | test
UTC       = lambda: datetime.now(timezone.utc).isoformat()
MM_DIR    = "/tmp/mm"
LL_DIR    = "/tmp/llama_bin"
WORK_DIR  = "/kaggle/working"
PORT      = int(os.environ.get("PORT", "8080"))
CTX       = int(os.environ.get("CTX", "32768"))        # context length
results   = {"mode": MODE, "phases": {}, "started": UTC()}

def run(cmd, timeout=None, stdin_devnull=False):
    print("$ " + (" ".join(cmd) if isinstance(cmd, list) else cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL if stdin_devnull else None)

def phase(name, fn):
    t0 = time.time()
    print("\n" + "=" * 68 + f"\n### PHASE {name}\n" + "=" * 68, flush=True)
    try:
        r = fn()
        results["phases"][name] = {"ok": True, "sec": round(time.time() - t0, 1)}
        return r
    except Exception as e:
        import traceback
        results["phases"][name] = {"ok": False, "sec": round(time.time() - t0, 1),
                                   "err": str(e), "tb": traceback.format_exc()[-2000:]}
        print(f"[phase {name} FAILED] {e}\n{traceback.format_exc()[-2000:]}", flush=True)
        return None

# ---------------------------------------------------------------- A. recon
def phase_a():
    r = run(["bash", "-lc",
             "nvidia-smi; echo '---GPU LIST---'; nvidia-smi -L; "
             "echo '---MEM---'; free -g; echo '---CPU---'; nproc; "
             "echo '---DISK---'; df -h /tmp /kaggle/working; "
             "echo '---TOOLS---'; which curl wget git cmake"], timeout=180)
    out = r.stdout or ""
    print(out, flush=True)
    gpus = [l.strip() for l in out.splitlines() if re.match(r"^GPU \d+:", l)]
    m = re.search(r"Driver Version:\s*([\d.]+)", out)
    results["env"] = {"gpu_count": len(gpus), "gpu_list": gpus,
                      "driver": m.group(1) if m else None}
    print(f"[recon] {len(gpus)} GPU(s): {gpus}", flush=True)
    return len(gpus)

n_gpu = phase("A_recon", phase_a) or 0

# ------------------------------------------------- B. llama.cpp (prebuilt)
# Two tarballs: the binaries + the CUDA runtime libs. They must live in the
# SAME directory because the binaries use RUNPATH=$ORIGIN.
# DO NOT set LD_LIBRARY_PATH to that dir: the cudart package ships an NVML
# stub that shadows the driver's libnvidia-ml.so (-> CPU fallback + dead
# nvidia-smi).  This cost a whole debugging cycle; don't repeat it.
REPO_ID   = "b11583"
MAIN_TARBALL = (f"https://github.com/ggml-org/llama.cpp/releases/download/{REPO_ID}/"
                f"llama-{REPO_ID}-bin-ubuntu-cuda-12.8-x64.tar.gz")
CUDA_TARBALL = (f"https://github.com/ggml-org/llama.cpp/releases/download/{REPO_ID}/"
                f"cudart-llama-{REPO_ID}-bin-ubuntu-cuda-12.8-x64.tar.gz")

def phase_b():
    r = run(["bash", "-lc",
             f"set -e; mkdir -p {LL_DIR}; "
             f"curl -fL --retry 3 -o /tmp/llama_main.tar.gz '{MAIN_TARBALL}'; "
             f"curl -fL --retry 3 -o /tmp/llama_cuda.tar.gz '{CUDA_TARBALL}'; "
             f"tar -xzf /tmp/llama_main.tar.gz -C {LL_DIR}; "
             f"mkdir -p {LL_DIR}/__cuda__ && tar -xzf /tmp/llama_cuda.tar.gz -C {LL_DIR}/__cuda__; "
             f"LMDIR=$(dirname $(find {LL_DIR} -name llama-server | head -1)); "
             f"echo LMDIR=$LMDIR; cp {LL_DIR}/__cuda__/*/*.so* $LMDIR/ 2>/dev/null || true; "
             f"ls $LMDIR | head -40"], timeout=1800)
    print(r.stdout, (r.stderr or "")[-500:], flush=True)
    lm_dir = None
    for line in r.stdout.splitlines():
        if line.startswith("LMDIR="):
            lm_dir = line.split("=", 1)[1].strip()
    hits = glob.glob(f"{lm_dir}/llama-server") if lm_dir else []
    if not hits:
        hits = glob.glob(f"{LL_DIR}/**/llama-server", recursive=True)
    if not hits:
        raise RuntimeError("llama-server not found in tarball")
    lm_dir = os.path.dirname(hits[0])
    ver = run([os.path.join(lm_dir, "llama-server"), "--version"], timeout=120)
    print("[llama-server version]", ((ver.stdout or "") + (ver.stderr or "")).strip()[:200], flush=True)
    if ver.returncode != 0:
        raise RuntimeError("prebuilt binary not runnable (glibc?)")
    bins = {n: os.path.join(lm_dir, n) for n in ("llama-server", "llama-cli")
            if os.path.exists(os.path.join(lm_dir, n))}
    print(f"[binaries] {bins}", flush=True)
    return bins

bins = phase("B_llamacpp", phase_b) or {}

# ---------------------------------------------------------- C. model download
REPO  = "AtomicChat/MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-GGUF"
QUANT = "AD-Q4_K-IQ4_XS"   # smallest quant, ~20 GB across 11 parts

def phase_c():
    run(["bash", "-lc", "pip install -q -U huggingface_hub requests 2>&1 | tail -1"], timeout=300)
    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(REPO)
    parts  = sorted(f for f in files if QUANT in f and f.endswith(".gguf"))
    mmproj = [f for f in files if f.startswith("mmproj-") and f.endswith("-F16.gguf")]
    if not parts or not mmproj:
        raise RuntimeError(f"unexpected repo layout: parts={parts[:2]} mmproj={mmproj}")
    pats = [f"*{os.path.basename(p)}" for p in parts] + [mmproj[0]]
    t0 = time.time()
    snapshot_download(REPO, local_dir=MM_DIR, allow_patterns=pats, max_workers=8)
    for f in parts + mmproj:
        p = os.path.join(MM_DIR, f)
        print(f"[file] {p}  {os.path.getsize(p)/1e9:.2f} GB", flush=True)
    results["download_sec"] = round(time.time() - t0, 1)
    return parts + mmproj

model_files = phase("C_download", phase_c) or []
LM_PART1    = next((f for f in model_files if QUANT in f and "-00001-of-" in f), None)
MMPROJ_FILE = next((os.path.join(MM_DIR, f) for f in model_files
                    if os.path.basename(f).startswith("mmproj-")), None)

# ------------------------------------------------------------- split helper
def split_flags(n):
    # Pipeline-parallel layer split across all visible T4s (no NVLink; fine).
    if n >= 2:
        return ["-ngl", "999", "-sm", "layer", "-ts", ",".join(["1"] * n)]
    # Single-GPU fallback: keep some MoE expert tensors on CPU, auto-fit the rest.
    return ["-ngl", "999", "-sm", "layer", "--cpu-moe", "--n-cpu-moe", "24"]

SERVER_BASE = (f"http://127.0.0.1:{PORT}")

def start_server():
    """Start llama-server, retrying with safer flags. Returns Popen or None."""
    import requests
    if not bins.get("llama-server") or not LM_PART1:
        print("[server] binaries or model missing", flush=True)
        return None
    base = [bins["llama-server"], "-m", os.path.join(MM_DIR, LM_PART1),
            "--mmproj", MMPROJ_FILE,
            "-c", str(CTX), "-t", "4",
            "--host", "0.0.0.0", "--port", str(PORT)]
    attempts = [
        base + split_flags(n_gpu) + ["-ctk", "q8_0", "-ctv", "q8_0"],
        base + split_flags(n_gpu) + ["-ctk", "f16", "-ctv", "f16"],
        base + split_flags(n_gpu),
    ]
    for i, cmd in enumerate(attempts):
        print(f"\n[server start attempt {i+1}/{len(attempts)}]\n$ {' '.join(cmd[1:])}", flush=True)
        subprocess.run(["bash", "-lc", "pkill -f llama-server || true"], timeout=30)
        lf = open("/tmp/server.log", "w")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid)
        t0, healthy = time.time(), False
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                break
            try:
                if requests.get(SERVER_BASE + "/health", timeout=5).status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(5)
        lf.close()
        print(f"[server] healthy={healthy} after {round(time.time()-t0,1)}s", flush=True)
        if healthy:
            results["server_cmd"] = " ".join(cmd[1:])
            results["server_load_sec"] = round(time.time() - t0, 1)
            return proc
        print("[server start failed] log tail:\n" + open("/tmp/server.log").read()[-2500:], flush=True)
    return None

proc = phase("D_server", start_server)

# ---------------------------------------------------------------- self test
def chat(messages, max_tokens=512, temperature=0.6, timeout=900):
    import requests
    t0 = time.time()
    r = requests.post(SERVER_BASE + "/v1/chat/completions",
                      json={"model": "mm", "messages": messages,
                            "max_tokens": max_tokens, "temperature": temperature},
                      timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    ct = j.get("usage", {}).get("completion_tokens", 0)
    # NOTE: MiniCPM-V 4.7 is a *thinking* model: reasoning comes back in
    # `reasoning_content`, the spoken answer in `content`.
    return {"content": msg.get("content", ""),
            "reasoning": msg.get("reasoning_content", ""),
            "completion_tokens": ct, "wall_sec": round(dt, 1),
            "tok_per_s": round(ct / dt, 2) if dt > 0 else None,
            "finish_reason": j["choices"][0].get("finish_reason")}

def gpu_snapshot():
    r = run(["bash", "-lc",
             "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
             "utilization.gpu --format=csv,noheader"], timeout=60)
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]

def phase_selftest():
    if proc is None:
        raise RuntimeError("no server running")
    results["gpu_mem_after_load"] = gpu_snapshot()
    print("[nvidia-smi per-GPU]\n" + "\n".join(results["gpu_mem_after_load"]), flush=True)

    print("[warmup] first request pays a one-time CUDA init cost...", flush=True)
    t0 = time.time()
    w = chat([{"role": "user", "content": "Say hi."}], max_tokens=8)
    print(f"[warmup] {round(time.time()-t0,1)}s -> {w['content'][:60]!r}", flush=True)
    results["warmup"] = w

    r1 = chat([{"role": "user",
                "content": "Explain mixture-of-experts neural networks in one sentence."}],
              max_tokens=256)
    print("[text test]", json.dumps(r1, indent=1)[:1200], flush=True)
    results["text_test"] = r1

    r2 = chat([{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url":
             "https://huggingface.co/datasets/huggingface/documentation-images/"
             "resolve/main/p-blog/candy.JPG"}},
        {"type": "text", "text": "What animal is on the candy? Answer shortly."}]}])
    print("[vision test]", json.dumps(r2, indent=1)[:1200], flush=True)
    results["vision_test"] = r2
    results["gpu_mem_final"] = gpu_snapshot()

if proc is not None:
    phase("E_selftest", phase_selftest)

# ------------------------------------------------------------ write status
def write_status():
    os.makedirs(WORK_DIR, exist_ok=True)
    info = {
        "service": "llama.cpp OpenAI-compatible server",
        "model": REPO + " (" + QUANT + ")",
        "base_url_in_session": SERVER_BASE,
        "endpoints": {
            "chat": SERVER_BASE + "/v1/chat/completions",
            "health": SERVER_BASE + "/health",
            "models": SERVER_BASE + "/v1/models",
        },
        "gpus": results.get("env", {}).get("gpu_list"),
        "gpu_memory_after_load": results.get("gpu_mem_after_load"),
        "context": CTX,
        "kv_cache": "q8_0",
        "server_cmd": results.get("server_cmd"),
        "updated": UTC(),
    }
    with open(os.path.join(WORK_DIR, "server_info.json"), "w") as f:
        json.dump(info, f, indent=2)

write_status()

# ---------------------------------------------------------------- server loop
if MODE == "server" and proc is not None:
    print("\n" + "=" * 68, flush=True)
    print("### SERVER READY - llama.cpp OpenAI API is serving")
    print(f"###   base_url : {SERVER_BASE}")
    print("###   chat     : POST /v1/chat/completions")
    print("###   from another notebook cell / terminal in this session:")
    print(f'###     curl {SERVER_BASE}/v1/chat/completions -H "Content-Type: application/json" \\\n'
          '###       -d \'{"model":"mm","messages":[{"role":"user","content":"hello"}],"max_tokens":128}\'')
    print("###   (Kaggle has no official tunnel: access is session-local unless")
    print("###    you bring your own connector - see repo README)")
    print("=" * 68 + "\n", flush=True)
    restarts, beat = 0, 0
    try:
        while True:
            time.sleep(300)                       # 5-minute heartbeat
            beat += 1
            try:
                import requests
                alive = requests.get(SERVER_BASE + "/health", timeout=5).status_code == 200
            except Exception:
                alive = False
            if proc.poll() is not None or not alive:
                restarts += 1
                print(f"[keepalive] server not answering (beat {beat}) - restarting...", flush=True)
                results.setdefault("restarts", 0)
                results["restarts"] = results.get("restarts", 0) + 1
                proc = start_server()
                if proc is None:
                    print("[keepalive] restart failed; will retry next beat", flush=True)
                    time.sleep(120)
            if beat % 6 == 0:                       # every ~30 min
                snap = gpu_snapshot()
                print(f"[keepalive] beat {beat} alive | " + " | ".join(snap), flush=True)
                write_status()
    except (KeyboardInterrupt, SystemExit):
        print("[keepalive] interrupted - shutting server down", flush=True)
        subprocess.run(["bash", "-lc", "pkill -f llama-server || true"], timeout=30)

results["finished"] = UTC()
with open(os.path.join(WORK_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\n### SUMMARY\n" + json.dumps(
    {k: v for k, v in results.items() if k in
     ("mode", "env", "server_load_sec", "gpu_mem_after_load", "gpu_mem_final",
      "phases", "text_test", "vision_test")}, indent=2)[:3500], flush=True)
print("DONE", flush=True)
