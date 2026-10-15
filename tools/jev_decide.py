#!/usr/bin/env python3
"""jev_decide.py — JEV-35B (AutoTrust System 1 decision model) on llama.cpp.

Replicates the System 1 read-out of the model's serve_decide.py against a plain
llama.cpp llama-server:

    prompt   = "[kind] {kind}\\n[state] {state}\\n[question] {question}"
               "\\n[options]\\nA) ...\\n[decision]:"
    read-out = softmax((logprob[label] + bias) / temperature)  over the kind's slots

The decision head is merged into the lm_head rows of the verbalizer tokens
(decision_head.json: "merged": true), so a plain text completion exposes it.

llama.cpp has no per-token-id logprob endpoint, so we bias all label token ids
by +100 (logit_bias) and read the top-N logprobs: the constant cancels in the
softmax, the non-label mass is negligible, and the returned label logprobs are
exact up to that vanishing term.

Usage:
    python3 jev_decide.py --model-dir /path/to/JEV-35B --port 8902 \
        --kind choice --question "Which city is the capital of France?" \
        --options "Paris" "London" "Berlin" "Madrid"

    # or as a server: POST /decide {"kind","state","question","options"}
    python3 jev_decide.py --model-dir /path/to/JEV-35B --port 8902 --serve 8940
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOP_K = 64  # top-N full-vocab logprobs requested; labels below this rank have negligible probability


def http_json(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class JevHead:
    """Slot layout, verbalizer token ids, bias and per-kind temperature."""

    def __init__(self, model_dir):
        with open(f"{model_dir}/adapter_vllm/decision_head.json") as f:
            head = json.load(f)
        with open(f"{model_dir}/calibration.json") as f:
            cal = json.load(f)
        self.ids = head["verbalizer_ids"]            # 264 token ids
        self.bias = head["bias"]                     # 264 biases
        self.slots = head["slots"]["ranges"]         # {"noul": [0,2], "score": [2,8], "choice": [8,264]}
        self.verbalizers = head["slots"]["verbalizers"]
        self.T = cal["per_kind"]

    def labels(self, kind, n_options):
        lo, hi = self.slots[kind]
        if kind == "choice":
            n = min(n_options, hi - lo)
            return list(range(lo, lo + n))
        return list(range(lo, hi))

    def prompt(self, kind, state, question, option_texts):
        lo = self.slots[kind][0]
        if kind == "choice":
            lines = [f"{self.verbalizers[lo + i]}) {t}" for i, t in enumerate(option_texts)]
        else:
            # serve_decide.py: noul and score have fixed option lines ("false","true" / "0".."5")
            lines = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return (f"[kind] {kind}\n[state] {state}\n[question] {question}\n"
                f"[options]\n" + "\n".join(lines) + "\n[decision]:")

    def decide(self, base_url, kind, state, question, option_texts):
        slots = self.labels(kind, len(option_texts or []))
        if kind != "choice":
            option_texts = None  # the option lines are fixed; slots cover the whole range
        ids = [self.ids[s] for s in slots]
        bias = [self.bias[s] for s in slots]
        # llama.cpp reports pre-bias full-vocab logprobs, so a plain top-N read is exact for
        # the labels it covers; "logprobs" is the integer form (n_probs), not the OpenAI bool
        body = {
            "prompt": self.prompt(kind, state, question, option_texts),
            "max_tokens": 1,
            "temperature": 1.0,       # full distribution, like serve_decide.py's READ
            "top_p": 1.0, "top_k": 0, "min_p": 0.0,
            "logprobs": TOP_K,        # a label outside the top TOP_K has negligible probability
        }
        out = http_json(f"{base_url}/v1/completions", body)
        lps = {t["id"]: t["logprob"] for t in out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        t = self.T.get(kind, 1.0)
        zs = [(lps.get(i, -1e9) + b) / t for i, b in zip(ids, bias)]
        m = max(zs)
        es = [math.exp(z - m) for z in zs]
        z = sum(es)
        probs = [e / z for e in es]
        return [{"label": self.verbalizers[s], "prob": p, "text": (option_texts or [""])[i] if i < len(option_texts or []) else ""}
                for i, (s, p) in enumerate(zip(slots, probs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="the JEV-35B checkout (decision_head.json, calibration.json)")
    ap.add_argument("--port", type=int, default=8080, help="llama-server port")
    ap.add_argument("--kind", choices=["choice", "score", "noul"], default="choice")
    ap.add_argument("--state", default="", help="context shown under [state]")
    ap.add_argument("--question", required=True)
    ap.add_argument("--options", nargs="*", default=[], help="choice: option texts; score: level descriptions")
    ap.add_argument("--serve", type=int, metavar="PORT", help="run an HTTP /decide gateway on this port instead")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    head = JevHead(args.model_dir)

    if not args.serve:
        t0 = time.time()
        result = head.decide(base, args.kind, args.state, args.question, args.options)
        dt = (time.time() - t0) * 1000
        top = max(result, key=lambda r: r["prob"])
        print(json.dumps({"answer": top["label"], "text": top["text"], "prob": round(top["prob"], 4),
                          "latency_ms": round(dt, 1), "distribution": result},
                         ensure_ascii=False, indent=2))
        return

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/decide":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            t0 = time.time()
            result = head.decide(base, req.get("kind", "choice"), req.get("state", ""),
                                 req["question"], req.get("options") or [])
            top = max(result, key=lambda r: r["prob"])
            body = json.dumps({"answer": top["label"], "prob": top["prob"], "distribution": result,
                               "latency_ms": round((time.time() - t0) * 1000, 1)},
                              ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"jev_decide gateway on http://0.0.0.0:{args.serve}/decide -> llama-server :{args.port}", file=sys.stderr)
    ThreadingHTTPServer(("0.0.0.0", args.serve), Handler).serve_forever()


if __name__ == "__main__":
    main()
