# MiniCPM-V 4.7 35B-A3B on Kaggle Dual Tesla T4

Serve **MiniCPM-V-4.7-35B-A3B** (GGUF quant) as an OpenAI-compatible vision+text
server on **Kaggle's free 2× Tesla T4 (2×16 GB)** — one push, runs unattended
for up to 12 hours, both GPUs used, no secrets anywhere in this repo.

> Measured on the real hardware (not vendor claims):
> **prefill ≈ 110 tok/s · decode ≈ 50–55 tok/s · vision ≈ 31 tok/s**,
> 21.3 GB of the 30 GB combined VRAM used.

Everything in this repo was executed end-to-end against the live Kaggle API and
the model repo; the benchmark numbers in [`results/benchmark.json`](results/benchmark.json)
were pulled from a real run's logs.

---

## What you get

| | |
|---|---|
| Model | [`AtomicChat/MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-GGUF`](https://huggingface.co/AtomicChat/MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-GGUF) (Q4_K-IQ4_XS, ~20 GB, 11 parts) |
| Runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) build `b11583` (CUDA 12.8 prebuilt + cudart libs) |
| Hardware | Kaggle `GPU T4 x2` — 2 × 15360 MiB, driver 580.x, 4 CPU, ~29–31 GB RAM |
| Features | text + vision (mmproj), 32K context, reasoning-aware, dual-GPU layer split |
| Runs | ~9 min to full speed; stays up to 12 h (Kaggle hard cap) |

## Repo layout

```
kernel/
  serve-minicpmv47-t4x2.py   # the Kaggle script (server mode by default)
  kernel-metadata.json       # Kaggle kernel metadata (GPU + internet ON)
scripts/
  push_kernel.sh             # one-command push to Kaggle (handles KGAT auth)
results/
  benchmark.json             # measured speeds + GPU memory from a real run
README.md
LICENSE                      # Apache-2.0 (matches the model license)
```

---

## Quick start (Kaggle, ~5 minutes)

**0. Prerequisites**

- Kaggle account **phone-verified** (Settings → Phone verification). Without this,
  GPU accelerators are greyed out.
- A Kaggle **API token** (Settings → API → *Create New Token* — the new-style
  `KGAT_...` tokens; see [auth note](#the-kgat-auth-trap-read-this-if-push-fails-with-401)).
- Installed: `pip install kaggle` (tested with 1.7.4.5).

**1. Push the kernel**

```bash
KAGGLE_API_TOKEN='KGAT_xxxxxxxx...' bash scripts/push_kernel.sh
```

That's it. Kaggle runs the kernel immediately: installs llama.cpp, downloads the
model (~90 s), loads it onto **both** T4s, runs a self-test, starts a Cloudflare
quick tunnel, and **publishes the public URL to ntfy.sh** (see step 2). Watch it:

```bash
KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN" \
  kaggle kernels status ajithkumaevnr2001/minicpmv47-server-dual-t4
# or open https://www.kaggle.com/code/<your-username>/minicpmv47-server-dual-t4
```

The kernel log prints `### SERVER ONLINE ###` with the URL when the API is live,
and the URL lands on `ntfy.sh/ajith-kaggle-minicpmv47-cluster` within seconds.

**2. Get the tunnel URL** (it changes every deployment — by design)

The kernel publishes its public HTTPS endpoint to an **ntfy.sh** topic the moment
the tunnel is up:

```bash
# latest URL (this is the discovery mechanism — poll it any time)
curl -s "https://ntfy.sh/ajith-kaggle-minicpmv47-cluster/json?poll=1&since=all" | tail -1

# subscribe live (browser/app):  https://ntfy.sh/ajith-kaggle-minicpmv47-cluster
# control topic (remote shutdown): https://ntfy.sh/ajith-kaggle-minicpmv47-control
```

The URL is also printed in the kernel log (`### SERVER ONLINE ###`) and written to
`/kaggle/working/tunnel_url.txt` + `server_info.json` (fetchable with
`kaggle kernels output <ref> -p out/` once the run ends).
It looks like: `https://<random-words>.trycloudflare.com`.

**3. Use it — UI or API**

- **Web UI (browser):** open the tunnel URL directly — llama.cpp's built-in
  WebUI is served at `/`.
- **API (OpenAI-compatible):** `base_url = <tunnel-url>/v1`

```python
# Official OpenAI Python SDK (pip install openai)
from openai import OpenAI
client = OpenAI(base_url="https://<random-words>.trycloudflare.com/v1", api_key="none")

# text
r = client.chat.completions.create(
    model="mm",
    messages=[{"role": "user", "content": "Explain mixture-of-experts in one sentence."}],
    max_tokens=512,
)
print(r.choices[0].message.content)
# NOTE: MiniCPM-V 4.7 is a *thinking* model - short answers may come back in
# message.reasoning_content with content empty; give it >=384-512 tokens.

# vision
r = client.chat.completions.create(
    model="mm",
    messages=[{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"}},
        {"type": "text", "text": "What animal is on the candy?"}]}],
    max_tokens=512,
)
```

```bash
curl https://<random-words>.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mm","messages":[{"role":"user","content":"hello"}],"max_tokens":512}'
```

Any OpenAI-compatible client works: Open WebUI, LobeChat, LangChain, LiteLLM,
Cursor — point them at `<tunnel-url>/v1`.

**4. Stop it (save GPU quota)**

```bash
# publish the control word; the kernel stops the server + tunnel and exits
curl -d "shutdown" https://ntfy.sh/ajith-kaggle-minicpmv47-control
```

> ⚠️ **Kaggle terms:** Kaggle does not officially support tunnels/ingress
> (cloudflared, ngrok, etc.). You asked for this explicitly and understand the
> tradeoff; usage is at your own discretion and risk. Sessions are also capped
> at 12 h and ~60 min idle; the kernel's heartbeat loop keeps it *active*.

---

## The script

`kernel/serve-minicpmv47-t4x2.py` runs five phases and then keeps the server alive:

| Phase | What it does |
|---|---|
| A recon | counts GPUs, driver, disk; prints to log |
| B llama.cpp | downloads **two** tarballs (binaries + CUDA runtime libs) and merges them into one dir |
| C download | pulls the multi-part GGUF + F16 mmproj from HF (~90 s, public repo, no token) |
| D server | starts `llama-server` split across both T4s, retries with safer flags |
| E self-test | warmup + text + vision requests with timing; writes `results.json` + `server_info.json` |
| keepalive | 5-min heartbeat; auto-restarts the server if it dies; logs GPU state every 30 min |

Environment knobs (set before pushing, they're read at run time):

| Var | Default | Meaning |
|---|---|---|
| `MODE` | `server` | `server` = serve + tunnel + keepalive; `test` = benchmark once and exit |
| `CTX` | `262144` | context length (tokens) — the model's native max. Auto-halves (128K→64K→32K) if the KV cache won't fit in VRAM at load time |
| `PORT` | `8080` | server port (the tunnel targets it) |
| `NTFY_TOPIC` | `ajith-kaggle-minicpmv47-cluster` | topic the public URL is published to |
| `NTFY_CONTROL` | `ajith-kaggle-minicpmv47-control` | topic that accepts the `shutdown` control word |
| `LLAMA_BUILD` | `b11583` | llama.cpp build to download |

The core launch command (proven to work; auto-derived from GPU count):

```bash
llama-server \
  -m MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-AD-Q4_K-IQ4_XS-00001-of-00011.gguf \
  --mmproj mmproj-MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-F16.gguf \
  -c 32768 -t 4 --host 0.0.0.0 --port 8080 \
  -ngl 999 -sm layer -ts 1,1 -ctk q8_0 -ctv q8_0
```

- `-ngl 999` → offload every layer that fits
- `-sm layer -ts 1,1` → pipeline-parallel layer split across the two T4s (no NVLink; transfers go over PCIe, ~fine at these sizes)
- `-ctk q8_0 -ctv q8_0` → 8-bit KV cache, halves context memory
- `--mmproj ...-F16.gguf` → vision tower (do **not** pick the BF16 file by substring match — see pitfalls)

## Measured performance (dual T4, Q4, 32K ctx)

Source: real run logs (`slot print_timing` + API wall-clock). Full payload in
[`results/benchmark.json`](results/benchmark.json).

| Metric | Value |
|---|---|
| Prompt prefill (675-token prompt incl. image) | **111.8 tokens/s** |
| Decode, 512 tokens (server timing) | **57.5 tokens/s** |
| Text chat via API (end-to-end) | **51.4–54.7 tokens/s** |
| Vision chat via API (image prefill + decode) | **31.0 tokens/s** |
| **Through the public Cloudflare tunnel (verified from outside Kaggle)** | **~47 tokens/s** steady; TTFT ≈ 0.5–1 s |
| GPU memory after load | **GPU0 11.42 GB · GPU1 9.83 GB** |
| Model load | ~85 s |
| Model download | ~90 s |
| First request after load | +40 s one-off (CUDA init + expert JIT; don't benchmark it) |

For reference on the same hardware: CPU-only (no offload) gives ~5 tok/s decode —
so the dual-GPU split is worth roughly **10×**.

## Tuning

**Context length.** The model is natively trained to 262,144 tokens (interleaved
linear/full attention makes KV tiny — only the 10 full-attention layers cache).
32K is a safe daily driver; **256K (the model's native max) has been verified running
on dual T4** — 21 GB of model + ~5 GB q8-KV fit the 30 GB combined VRAM. Beyond
memory, **prefill time** becomes the limit: a full 256K prompt takes minutes to
prefill, while decode stays ~50 tok/s. The script auto-halves the context
(128K→64K→32K) if the KV cache won't fit at load time. Set with `CTX=...`.

**Quantization.** Quants in the source repo:

| Quant | Size | Note |
|---|---|---|
| `AD-Q4_K-IQ4_XS` (default here) | ~20 GB | best VRAM/speed trade-off for dual T4 |
| `AD-Q5_K-Q4_K` | ~24 GB | higher quality, still fits |
| `AD-Q6_K` | ~28 GB | tight with KV on dual T4 |
| `AD-Q4_K` / larger | >24 GB | not recommended here |

Change `QUANT` at the top of the script (the phase-C filter auto-discovers files).

**KV cache.** `-ctk q8_0 -ctv q8_0` (default) halves KV vs f16. For long contexts
also consider `CTX` + q8; f16 fallback is built in.

**Threads.** `-t 4` matches Kaggle's 4 CPU cores (MoE decode is bandwidth-bound;
more threads don't help).

---

## Pitfalls that will cost you a day if you don't know them

1. **`llama-bench` ignores multi-GPU splits** (at least at `b11583`): it tries to
   allocate the *whole* model on `CUDA0` and OOMs. Measure speed with `llama-cli`
   (raw) or the server's `slot print_timing` / API round-trip instead.
2. **Never set `LD_LIBRARY_PATH` to the llama dir.** The `cudart-*` package ships
   an NVML stub (`libnvidia-ml.so`) that shadows the driver's library → llama.cpp
   silently falls back to **CPU** (5 tok/s) and `nvidia-smi` starts failing with
   "couldn't find libnvidia-ml.so". The binaries use `RUNPATH=$ORIGIN`, so no
   env var is needed.
3. **Two tarballs, one directory.** `llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz`
   has the binaries (flat `llama-b<tag>/` dir); `cudart-llama-<tag>-bin-...tar.gz`
   has `libcudart/libcublas` — copy the libs *next to* the binaries.
4. **BF16 mmproj matches an `"F16" in name` filter** — `mmproj-...-BF16.gguf`
   contains the substring `F16`. Filter with `endswith("-F16.gguf")` or you'll
   download the wrong file.
5. **It's a thinking model.** Answers can land in `reasoning_content` with
   `content` empty (especially with small `max_tokens`). Read both fields; give it
   ≥256 tokens for text, ≥512 when an image is attached.
6. **Multi-part GGUF:** pass only the *first* part (`...-00001-of-00011.gguf`);
   llama.cpp loads the rest automatically.
7. **First request is slow** (~40 s one-off cost). Always warm up before timing.

## The KGAT auth trap (read this if push fails with 401)

The current `kaggle` CLI (1.7.x) authenticates with **basic auth**
(`username` + key). Modern `KGAT_...` API tokens are **bearer-only** — so
`KAGGLE_USERNAME=... KAGGLE_KEY=KGAT_...` returns `401 Unauthorized` even with
the correct username.

The CLI's underlying `kagglesdk` client, however, still honors a bearer token via
the `KAGGLE_API_TOKEN` environment variable. So:

```bash
export KAGGLE_USERNAME=<your-kaggle-username>   # any non-empty value; not used
export KAGGLE_KEY=placeholder                    # only satisfies CLI validation
export KAGGLE_API_TOKEN='KGAT_xxxxxxxx...'       # the real credential
kaggle kernels push -p kernel
```

`scripts/push_kernel.sh` does exactly this. Verify auth with
`kaggle kernels list --mine` (should list your kernels, not 401).

## Automation

```bash
# push a new version (also triggers an immediate run)
bash scripts/push_kernel.sh

# status / logs / outputs
kaggle kernels status ajithkumaevnr2001/minicpmv47-server-dual-t4
kaggle kernels output ajithkumaevnr2001/minicpmv47-server-dual-t4 -p out/
```

Kaggle limits: 30 GPU-h/week, ≤12 h per session, 20 GB persisted to
`/kaggle/working`, ~60 min idle timeout. A no-op keepalive cell keeps a session
alive; this script's heartbeat loop keeps it *running* (not idle) for the full 12 h.

## Running outside Kaggle

The script only assumes: Linux, `nvidia-smi`, `curl`, `python3`, ≥2 GPUs with ≥15 GB
each (or 1 GPU → it auto-switches to CPU-expert offload). Change nothing else.

## Credits & licensing

- Model: [MiniCPM-V 4.7](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) by
  OpenBMB (Apache-2.0). Abliterated GGUF quant by
  [AtomicChat](https://huggingface.co/AtomicChat/MiniCPM-V-4.7-35B-A3B-Abliterated-Uncensored-GGUF).
  Respect the model card and intended-use terms.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT.
- This repo — Apache-2.0 (see [LICENSE](LICENSE)).
