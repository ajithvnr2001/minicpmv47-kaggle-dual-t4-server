# =====================================================================
#  MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-GGUF
#  Long-running dual-GPU (Kaggle T4 x2) llama.cpp server + Cloudflare tunnel
#
#  MODES (env MODE):
#    server (default) - install, download, start llama-server on BOTH T4s,
#                       self-test, expose it via a Cloudflare quick tunnel,
#                       publish the URL to ntfy.sh, and stay alive until Kaggle
#                       ends the session (12h cap). A control topic allows
#                       remote shutdown to save GPU quota.
#    test            - install, download, benchmark + functional tests, exit.
#
#  NO SECRETS: the HF repo is public. Nothing secret is written to log/ntfy.
# =====================================================================
import os, re, sys, time, json, glob, subprocess, threading
from datetime import datetime, timezone

MODE      = os.environ.get("MODE", "server").lower()   # server | test
NTFY_TOPIC   = os.environ.get("NTFY_TOPIC",   "ajith-kaggle-minicpmv47-cluster")
NTFY_CONTROL = os.environ.get("NTFY_CONTROL", "ajith-kaggle-minicpmv47-control")
UTC       = lambda: datetime.now(timezone.utc).isoformat()
MM_DIR    = "/tmp/mm"
LL_DIR    = "/tmp/llama_bin"
CF_BIN    = "/tmp/cloudflared"
WORK_DIR  = "/kaggle/working"
PORT      = int(os.environ.get("PORT", "8080"))
CTX       = int(os.environ.get("CTX", "32768"))
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
# Two tarballs: binaries + CUDA runtime libs, merged into ONE directory because
# the binaries use RUNPATH=$ORIGIN. DO NOT set LD_LIBRARY_PATH to that dir:
# the cudart package ships an NVML stub that shadows the driver's
# libnvidia-ml.so -> silent CPU fallback (5 tok/s) and dead nvidia-smi.
REPO_ID   = os.environ.get("LLAMA_BUILD", "b11583")
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
    results["llama_build"] = REPO_ID
    return bins

def phase_b_cloudflared():
    r = run(["bash", "-lc",
             f"curl -fL --retry 3 -o {CF_BIN} "
             "https://github.com/cloudflare/cloudflared/releases/latest/download/"
             f"cloudflared-linux-amd64 && chmod +x {CF_BIN} && {CF_BIN} --version"], timeout=600)
    print((r.stdout or "")[:200], (r.stderr or "")[-300:], flush=True)
    if not os.path.exists(CF_BIN) or r.returncode != 0:
        raise RuntimeError("cloudflared install failed")
    os.chmod(CF_BIN, 0o755)          # belt & braces (the earlier f-string bug bit here)
    print("[+] cloudflared installed", flush=True)
    return True

if MODE == "server":
    phase("B2_cloudflared", phase_b_cloudflared)

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
    if n >= 2:
        return ["-ngl", "999", "-sm", "layer", "-ts", ",".join(["1"] * n)]
    return ["-ngl", "999", "-sm", "layer", "--cpu-moe", "--n-cpu-moe", "24"]

SERVER_BASE = f"http://127.0.0.1:{PORT}"

def start_server():
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

# ------------------------------------------------------- cloudflare tunnel
URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

def start_tunnel():
    """Start cloudflared quick tunnel; return (proc, url|None). Never raises."""
    try:
        if not os.path.exists(CF_BIN):
            print("[tunnel] cloudflared missing", flush=True)
            return None, None
        os.chmod(CF_BIN, 0o755)
        subprocess.run(["bash", "-lc", "pkill -f cloudflared || true"], timeout=30)
        tf = open("/tmp/tunnel.log", "w")
        proc = subprocess.Popen([CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
                                stdout=tf, stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid)
    except Exception as e:
        print(f"[tunnel] start failed: {e}", flush=True)
        return None, None
    url, t0 = None, time.time()
    while time.time() - t0 < 120:
        if proc.poll() is not None:
            break
        try:
            m = URL_RE.search(open("/tmp/tunnel.log").read())
        except Exception:
            m = None
        if m:
            url = m.group(0)
            break
        time.sleep(2)
    tf.close()
    print(f"[tunnel] url={url} (elapsed {round(time.time()-t0,1)}s)", flush=True)
    if url is None:
        print("[tunnel] log tail:\n" + open("/tmp/tunnel.log").read()[-1500:], flush=True)
    return proc, url

def announce(url):
    """Publish the public endpoint: ntfy topic + files + log banner."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=url.encode("utf-8"),
            headers={"Title": "MiniCPM-V 4.7 Kaggle dual-T4 server live",
                     "Priority": "high", "Tags": "zap,gpu"})
        urllib.request.urlopen(req, timeout=10)
        print(f"[+] URL published to ntfy.sh/{NTFY_TOPIC}", flush=True)
    except Exception as e:
        print(f"[ntfy] announce failed: {e}", flush=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(os.path.join(WORK_DIR, "tunnel_url.txt"), "w") as f:
        f.write(url + "\n")
    info = {
        "service": "llama.cpp OpenAI-compatible server (MiniCPM-V 4.7)",
        "model": REPO + " (" + QUANT + ")",
        "public_base_url": url,
        "openai_base_url": url + "/v1",
        "web_ui_url": url + "/",
        "health": url + "/health",
        "endpoints_in_session": {"chat": SERVER_BASE + "/v1/chat/completions"},
        "gpus": results.get("env", {}).get("gpu_list"),
        "gpu_memory_after_load": results.get("gpu_mem_after_load"),
        "context": CTX, "kv_cache": "q8_0",
        "server_cmd": results.get("server_cmd"),
        "note": ("OpenAI-compatible: use <public_base_url>/v1 as base_url in any "
                 "OpenAI SDK / Open WebUI / LobeChat. Built-in llama.cpp WebUI at "
                 "<public_base_url>/."),
        "updated": UTC(),
    }
    with open(os.path.join(WORK_DIR, "server_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("\n" + "=" * 68)
    print("### SERVER ONLINE - MiniCPM-V 4.7 on Kaggle dual T4")
    print(f"###   OpenAI base URL : {url}/v1")
    print(f"###   Web UI          : {url}/")
    print(f"###   Health          : {url}/health")
    print(f"###   url also at     : {WORK_DIR}/tunnel_url.txt  and  ntfy.sh/{NTFY_TOPIC}")
    print("=" * 68 + "\n", flush=True)
    results["public_url"] = url

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
    # Thinking model: answer in `content`, reasoning in `reasoning_content`.
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
    print("[warmup] first request pays one-time CUDA init...", flush=True)
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

# ------------------------------------------------- tunnel + control + keepalive
def session_keepalive():
    """Fallback loop: keep the process alive until the Kaggle session ends."""
    print("[keepalive] server alive (session-local on :%d); no tunnel" % PORT, flush=True)
    while True:
        time.sleep(300)
        if proc is not None and proc.poll() is not None:
            print("[keepalive] server process died - restarting", flush=True)
            globals()["proc"] = start_server()
        else:
            print("[keepalive] beat | " + " | ".join(gpu_snapshot()), flush=True)

if MODE == "server" and proc is not None:
    try:
        tproc, turl = start_tunnel()
        if turl:
            results["tunnel_started"] = UTC()
            announce(turl)
        else:
            print("[!] tunnel did not come up; server stays session-local on :%d" % PORT, flush=True)

        # ---- remote shutdown listener (ntfy control topic) ----
        STOP_TS = int(time.time())
        def control_listener():
            while True:
                try:
                    import urllib.request
                    with urllib.request.urlopen(
                            f"https://ntfy.sh/{NTFY_CONTROL}/json?poll=1&since={STOP_TS}",
                            timeout=30) as resp:
                        for line in resp:
                            if b"shutdown" in line:
                                print("\n[!] REMOTE SHUTDOWN received - stopping server + tunnel", flush=True)
                                for p in (tproc,):
                                    try:
                                        p and p.terminate()
                                    except Exception:
                                        pass
                                subprocess.run(["bash", "-lc",
                                                "pkill -f llama-server; pkill -f cloudflared || true"],
                                               timeout=30)
                                os._exit(0)
                except Exception:
                    pass
                time.sleep(5)
        threading.Thread(target=control_listener, daemon=True).start()
        print(f"[+] remote shutdown armed: publish 'shutdown' to ntfy.sh/{NTFY_CONTROL}", flush=True)

        # ---- keepalive: restart server/tunnel if either dies ----
        beat = 0
        while True:
            time.sleep(300)
            beat += 1
            import requests
            srv_ok = False
            try:
                srv_ok = requests.get(SERVER_BASE + "/health", timeout=5).status_code == 200
            except Exception:
                pass
            tun_ok = tproc is not None and tproc.poll() is None
            if proc.poll() is not None or not srv_ok:
                print(f"[keepalive] server down (beat {beat}) - restarting...", flush=True)
                globals()["proc"] = start_server()
                if proc is None:
                    print("[keepalive] server restart failed; retry next beat", flush=True)
                    time.sleep(120)
            if tproc is not None and (tproc.poll() is not None or not tun_ok):
                print(f"[keepalive] tunnel down (beat {beat}) - restarting + republishing...", flush=True)
                tproc, turl = start_tunnel()
                if turl:
                    announce(turl)
            if beat % 6 == 0:
                print(f"[keepalive] beat {beat} | server={'up' if srv_ok else 'down'} "
                      f"tunnel={'up' if tun_ok else 'down'} | "
                      + " | ".join(gpu_snapshot()), flush=True)
    except Exception as e:
        # A tunnel problem must never take the serving session down.
        print(f"[tunnel block] unexpected error: {e} - falling back to session-local serving", flush=True)
        session_keepalive()
    except (KeyboardInterrupt, SystemExit):
        print("[keepalive] interrupted - shutting down", flush=True)
        subprocess.run(["bash", "-lc",
                        "pkill -f llama-server; pkill -f cloudflared || true"], timeout=30)

results["finished"] = UTC()
with open(os.path.join(WORK_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\n### SUMMARY\n" + json.dumps(
    {k: v for k, v in results.items() if k in
     ("mode", "env", "public_url", "server_load_sec", "gpu_mem_after_load",
      "gpu_mem_final", "phases", "text_test", "vision_test")}, indent=2)[:3500], flush=True)
print("DONE", flush=True)
