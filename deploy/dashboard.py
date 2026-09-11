"""Interactive J-lens dashboard. Runs ON the vast.ai instance only.

Loads one quantized model + one published lens once, then serves a prompt box
that renders jlens' own slice visualisation. Bound to 127.0.0.1 by bootstrap.sh
and reached through an SSH tunnel, so there is no auth, no multi-user handling
and no persistence here by design.

    uvicorn dashboard:app --host 127.0.0.1 --port 7860
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

MODEL_ID = os.environ.get("JLENS_MODEL_ID", "Qwen/Qwen3.8-27B")
LENS_REPO = os.environ.get("JLENS_LENS_REPO", "eyes-ml/Qwen3.8-27B_jacobian-lens")
LENS_FILE = os.environ.get("JLENS_LENS_FILE", "Qwen3.8-27B_jacobian_lens.pt")
PRECISION = os.environ.get("JLENS_PRECISION", "nf4")

# Hard caps. The page offers these as defaults; the server re-clamps because the
# form is trivially editable and a 512-token x stride-1 slice is a much bigger
# bill (two full-vocab argsort sweeps per row) than it looks in the UI.
MAX_SEQ_CAP = 512
TOP_N_CAP = 12
MAX_TRACKED = 256  # mandatory for mode="embed": the page inlines one rank array per tracked token
GEN_CAP = 128

EXAMPLES = [
    "Fact: The currency used in the country shaped like a boot is",
    "Fact: The capital city of the country that hosted the 2016 Summer Olympics is",
    "Fact: The element with atomic number 79, the one used in wedding rings, is called",
    "Fact: The author of the play in which the prince of Denmark sees his father's ghost was born in",
]

_lock = threading.Lock()  # jlens registers hooks on the shared blocks: one slice at a time
_state: dict = {"status": "loading", "detail": "starting", "error": None}
_model = _lens = _tokenizer = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load() -> None:
    """Load model + lens. Runs on a background thread so the page can serve a
    readable status (and a readable failure) during the ~15 min cold start."""
    global _model, _lens, _tokenizer
    try:
        import torch
        import transformers

        import jlens
        import jlens.vis as jvis

        _state["detail"] = f"loading {MODEL_ID} ({PRECISION})"
        log(_state["detail"])
        if PRECISION == "nf4":
            quant = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["lm_head"],
            )
        elif PRECISION == "int8":
            quant = transformers.BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=["lm_head"]
            )
        elif PRECISION == "bf16":
            quant = None
        else:
            raise ValueError(f"JLENS_PRECISION must be nf4|int8|bf16, got {PRECISION!r}")

        t0 = time.time()
        # AutoModelForImageTextToText, not AutoModelForCausalLM: this checkpoint is a VLM
        # and the CausalLM alias resolves to a wrapper jlens cannot find the decoder in.
        # device_map={"": 0} not "auto": "auto" would scatter layers and reinstate a
        # per-call host->device copy of every Jacobian.
        hf_model = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0}
        )
        _tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = jlens.from_hf(hf_model, _tokenizer, compile=False)
        log(f"model loaded in {time.time() - t0:.0f}s | {model!r} | layout={model.layout.path}")

        # unembed() casts residuals to the lm_head dtype. A quantized head would make
        # every readout silent garbage, so refuse to serve rather than mislead.
        head_dtype = model._lm_head.weight.dtype
        if head_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise RuntimeError(
                f"lm_head is {head_dtype}; it was quantized and every lens readout "
                "would be garbage. Check llm_int8_skip_modules."
            )

        _state["detail"] = f"loading lens {LENS_REPO}"
        log(_state["detail"])
        lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
        if lens.d_model != model.d_model:
            raise RuntimeError(f"lens d_model={lens.d_model} != model d_model={model.d_model}")
        bad = [n for n in lens.source_layers if not 0 <= n < model.n_layers]
        if bad:
            raise RuntimeError(f"lens layers {bad} out of range for {model.n_layers} layers")
        # Pin the Jacobians to the GPU once. transport() otherwise copies ~100MB per
        # layer per pass, twice per layer per request.
        lens.jacobians = {n: J.to(model.input_device) for n, J in lens.jacobians.items()}
        log(f"lens loaded | {lens!r}")

        _state["detail"] = "warming caches"
        log(_state["detail"])
        vocab = int(model._lm_head.weight.shape[0])
        # ~250k single-token decodes, cached for process lifetime. Paying it here keeps
        # it off the first request.
        jvis._meaningful_token_mask(_tokenizer, vocab, model.input_device)
        try:
            jvis._template("embed")  # one-off d3 fetch; embed mode needs it inlined
        except Exception as exc:  # noqa: BLE001 - non-fatal, retried per request
            log(f"WARNING: d3 prefetch failed ({exc}); /run retries before computing")

        peak = torch.cuda.max_memory_allocated() / 1024**3
        now = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"VRAM: {now:.1f} GiB resident, {peak:.1f} GiB peak, {total:.1f} GiB total")

        _model, _lens = model, lens
        _state.update(
            status="ready",
            detail=(
                f"{MODEL_ID} · {PRECISION} · {model.n_layers} layers · d_model {model.d_model} · "
                f"lens {len(lens.source_layers)} fitted layers · {peak:.1f}/{total:.0f} GiB VRAM peak"
            ),
        )
        log("ready")
    except Exception:
        _state.update(status="failed", detail="load failed", error=traceback.format_exc())
        log("LOAD FAILED\n" + _state["error"])


def _generate(context_token_ids: list[int], n_new: int, top_n: int) -> dict:
    """Greedy-decode ``n_new`` tokens from the exact ids the slice used, and
    return the continuation plus the top-``top_n`` next-token distribution at
    the last prompt position. This is what the L63 row is predicting, with
    probabilities, and unmasked (mask_display hides non-word tokens there)."""
    import torch

    hf = _model._hf_model
    ids = torch.tensor([context_token_ids], device=_model.input_device)
    out = hf.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=n_new,
        do_sample=False,
        pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    new_ids = out.sequences[0, ids.shape[1] :].tolist()
    probs = torch.softmax(out.logits[0][0].float(), dim=-1)
    top = probs.topk(top_n)
    return {
        "continuation": _tokenizer.decode(new_ids, skip_special_tokens=False),
        "next": [
            {"token": _tokenizer.decode([int(t)], clean_up_tokenization_spaces=False), "p": float(p)}
            for p, t in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/status")
def status() -> JSONResponse:
    return JSONResponse(
        {"status": _state["status"], "detail": _state["detail"], "error": _state["error"]}
    )


@app.post("/run")
def run(req: dict = Body(...)) -> JSONResponse:
    """Compute one slice and return the self-contained page as a string.

    The page goes back in JSON and the browser drops it into iframe.srcdoc. That
    beats a GET /slice/{id} route because it needs no server-side store, no ids
    and no eviction: mode="embed" pages are already fully self-contained, so the
    only thing a store would buy is a URL nobody asked for.
    """
    if _state["status"] != "ready":
        return JSONResponse(
            {"error": f"model is not ready ({_state['status']}: {_state['detail']})"}, 503
        )

    prompt = str(req.get("prompt", "")).strip()
    if not prompt:
        return JSONResponse({"error": "prompt is empty"}, 400)

    def clamp(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(req.get(key, default))))
        except (TypeError, ValueError):
            return default

    max_seq_len = clamp("max_seq_len", 256, 16, MAX_SEQ_CAP)
    layer_stride = clamp("layer_stride", 2, 1, 8)
    top_n = clamp("top_n", 8, 1, TOP_N_CAP)
    last_n = clamp("last_n_tokens", 0, 0, MAX_SEQ_CAP) or None
    gen_tokens = clamp("gen_tokens", 32, 0, GEN_CAP)

    import torch

    from jlens.vis import _template, build_page, compute_slice

    with _lock:
        try:
            # Retry the d3 fetch here, not inside build_page: vis memoises only on
            # success, so a failure there would land after the whole slice compute.
            _template("embed")
            t0 = time.time()
            data = compute_slice(
                _model,
                _lens,
                prompt,
                top_n=top_n,
                layer_stride=layer_stride,
                last_n_tokens=last_n,
                max_seq_len=max_seq_len,
                max_tracked=MAX_TRACKED,
                mask_display=True,
            )
            t_compute = time.time() - t0
            t0 = time.time()
            page, _raw, _payload = build_page(
                data,
                prompt,
                title="J-lens slice",
                description=f"{MODEL_ID} ({PRECISION}), lens {LENS_REPO}",
                mode="embed",
            )
            t_render = time.time() - t0
            t0 = time.time()
            # Outside compute_slice, so no lens hooks are registered; inside the
            # lock, since it shares the GPU. Runs on the ids the slice used, so
            # the continuation follows exactly the prompt the grid shows.
            output = _generate(data.context_token_ids, gen_tokens, top_n) if gen_tokens else None
            t_gen = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return JSONResponse(
                {"error": "CUDA out of memory. Lower max_seq_len or raise layer_stride, then retry."},
                500,
            )
        except Exception as exc:  # noqa: BLE001 - surface the real thing, do not hide it
            log("RUN FAILED\n" + traceback.format_exc())
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, 500)

    used = len(data.context_token_ids)
    stats = (
        f"{used} tokens used"
        + (f" (truncated at max_seq_len={max_seq_len})" if used == max_seq_len else "")
        + f" · {data.seq_len} positions x {len(data.layers)} rows"
        + f" · forward+readout {t_compute:.1f}s · render {t_render:.1f}s"
        + (f" · generate {t_gen:.1f}s" if output else "")
        + f" · page {len(page) / 1e6:.1f} MB"
    )
    log(stats)
    return JSONResponse({"html": page, "stats": stats, "prompt": prompt, "output": output})


PAGE = """<!doctype html><meta charset=utf-8><title>J-lens</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0d0f12;color:#c9cdd4;
     font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:10px 14px;border-bottom:1px solid #1e2229;display:flex;gap:14px;align-items:baseline}
h1{font-size:13px;margin:0;color:#e6e9ee;font-weight:600;letter-spacing:.04em}
#detail{color:#6b727d;font-size:11px}
main{padding:14px;display:flex;flex-direction:column;gap:10px}
textarea{width:100%;height:64px;resize:vertical;background:#12151a;color:#e6e9ee;
         border:1px solid #262b33;border-radius:3px;padding:8px;font:inherit}
textarea:focus,input:focus{outline:none;border-color:#3d6ea8}
.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
label{color:#6b727d;font-size:11px}
input[type=number]{width:66px;background:#12151a;color:#e6e9ee;border:1px solid #262b33;
                   border-radius:3px;padding:3px 5px;font:inherit}
button{background:#1b2531;color:#cfe0f5;border:1px solid #33465e;border-radius:3px;
       padding:5px 14px;font:inherit;cursor:pointer}
button:hover:enabled{background:#22303f}
button:disabled{opacity:.4;cursor:default}
.ex{background:none;border:none;color:#6f8db3;padding:0;text-align:left;font-size:11px;
    cursor:pointer;text-decoration:underline dotted}
.ex:hover{color:#9dc0e8}
#stats{color:#6b727d;font-size:11px;min-height:1.5em}
#err{white-space:pre-wrap;color:#e08a7a;background:#1a1214;border:1px solid #3d2226;
     border-radius:3px;padding:8px;font-size:11px;display:none}
#output{display:none;background:#12151a;border:1px solid #262b33;border-radius:3px;padding:8px 10px;
        white-space:pre-wrap;word-break:break-word}
#output .ctx{color:#6b727d}
#output .gen{color:#e6e9ee;background:#1b2531}
#next{color:#6b727d;font-size:11px;margin-top:6px}
#next b{color:#cfe0f5;font-weight:600}
iframe{width:100%;height:78vh;border:1px solid #262b33;border-radius:3px;background:#fff;display:none}
</style>
<header><h1>J-LENS</h1><span id=detail>connecting...</span></header>
<main>
<textarea id=prompt spellcheck=false placeholder="prompt"></textarea>
<div class=row id=examples></div>
<div class=row>
  <label>max_seq_len <input type=number id=max_seq_len value=256 min=16 max=512></label>
  <label>layer_stride <input type=number id=layer_stride value=2 min=1 max=8></label>
  <label>top_n <input type=number id=top_n value=8 min=1 max=12></label>
  <label>last_n_tokens <input type=number id=last_n_tokens value=0 min=0 max=512></label>
  <label>gen_tokens <input type=number id=gen_tokens value=32 min=0 max=128></label>
  <button id=go disabled>run</button>
  <span id=stats></span>
</div>
<div id=err></div>
<div id=output><span class=ctx></span><span class=gen></span><div id=next></div></div>
<iframe id=out></iframe>
</main>
<script>
const EX = __EXAMPLES__;
const $ = id => document.getElementById(id);
EX.forEach(t => {
  const b = document.createElement('button');
  b.className = 'ex'; b.textContent = t;
  b.onclick = () => { $('prompt').value = t; };
  $('examples').appendChild(b);
});
// The final row (L63) is the model's own output, not a lens readout: it is the
// sanity check that the NF4 quantization has not drifted the lens off.
let busy = false;
// Greedy continuation, highlighted after the dimmed prompt, then the top next-token
// probabilities at the last position: the distribution the L63 row is showing top-1 of.
function showOutput(prompt, o) {
  const box = $('output');
  if (!o) { box.style.display = 'none'; return; }
  box.querySelector('.ctx').textContent = prompt;
  box.querySelector('.gen').textContent = o.continuation;
  const next = $('next'); next.textContent = 'next token: ';
  o.next.forEach((n, i) => {
    if (i) next.append('  ');
    const b = document.createElement('b'); b.textContent = JSON.stringify(n.token);
    next.append(b, ' ' + (100 * n.p).toFixed(1) + '%');
  });
  box.style.display = 'block';
}
async function poll() {
  let delay = 3000;
  try {
    const s = await (await fetch('/status')).json();
    $('detail').textContent = s.status === 'ready' ? s.detail : s.status + ': ' + s.detail;
    if (!busy) $('go').disabled = s.status !== 'ready';
    if (s.status === 'failed') { $('err').style.display = 'block'; $('err').textContent = s.error; }
    if (s.status !== 'loading') delay = 15000;  // keep watching: bootstrap.sh is re-runnable
  } catch (e) { $('detail').textContent = 'server unreachable'; }
  setTimeout(poll, delay);
}
poll();
$('go').onclick = async () => {
  busy = true; $('go').disabled = true; $('err').style.display = 'none';
  $('stats').textContent = 'running...';
  const body = {prompt: $('prompt').value};
  for (const k of ['max_seq_len','layer_stride','top_n','last_n_tokens','gen_tokens'])
    body[k] = parseInt($(k).value, 10);
  try {
    const r = await fetch('/run', {method:'POST', headers:{'Content-Type':'application/json'},
                                   body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok || d.error) {
      $('stats').textContent = '';
      $('err').style.display = 'block'; $('err').textContent = d.error || ('HTTP ' + r.status);
    } else {
      $('stats').textContent = d.stats;
      showOutput(d.prompt, d.output);
      $('out').style.display = 'block'; $('out').srcdoc = d.html;
    }
  } catch (e) {
    $('stats').textContent = '';
    $('err').style.display = 'block'; $('err').textContent = String(e);
  }
  busy = false; $('go').disabled = false;
};
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # json.dumps, not html.escape: this lands inside a <script> block, where HTML
    # entities are not decoded. The </ guard keeps a literal from closing the tag.
    examples = json.dumps(EXAMPLES).replace("</", "<\\/")
    return HTMLResponse(PAGE.replace("__EXAMPLES__", examples))
