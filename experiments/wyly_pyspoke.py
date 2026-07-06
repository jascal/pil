"""Minimal HTTP spoke over the PYTHON reference runtime (exact by construction) -- used for the
composed-unit evaluation while the C++ spoke's Qwen-tokenizer parity is an open question.
Chain mode: greedy multi-token completion, conf-floor stop. OpenAI-shaped like sgiandubh."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "rosetta" / "py"))
from serve_package import decide, load_package  # noqa: E402

from pil.tokens import TokenSpace  # noqa: E402

PKG = Path(sys.argv[1])
PORT = int(sys.argv[2])
ts = TokenSpace.from_file(PKG / "bundle.tokenizer.json")
idioms, ngrams, m = load_package(PKG / "manifest.json")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = ""
        for msg in body.get("messages", []):
            if msg.get("role") == "user":
                q = msg.get("content", "")
        ctx = list(ts.encode(" " + q if not q.startswith(" ") else q))
        out, cite, route, minconf = [], "", "", 1e9
        for step in range(12):
            d = decide(ctx, idioms, ngrams, m)
            if d is None or (step > 0 and d.get("confidence", 1.0) < 0.30):
                break
            if step == 0:
                c = d.get("citation", "")
                cite = c if isinstance(c, str) else "; ".join(c)
                route = f"{d.get('tier')}/{d.get('basis')}"
            minconf = min(minconf, d.get("confidence", 1.0))
            ctx.append(d["answer"])
            out.append(d["answer"])
            if "." in ts.token_str(d["answer"]) and step > 0:
                break
        txt = ts.decode(out).strip() if out else ""
        a = ({"answer": txt, "kind": "distilled", "citation": cite, "route": route,
              "confidence": round(minconf, 4)} if txt else {"answer": "", "kind": "abstain"})
        resp = {"model": "pyspoke", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(a)}}]}
        b = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


HTTPServer(("0.0.0.0", PORT), H).serve_forever()
