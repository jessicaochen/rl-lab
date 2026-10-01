#!/usr/bin/env python3
"""Render readable transcripts from uni-agent's token-level trajectories.

Each session dir in runs/<id>/logs/agent-logs.tar.gz holds trajectory.json
(meta) + trajectory.npz (prompt_ids, response_ids, response_mask, logprobs).
This decodes them with the model tokenizer: response tokens with mask=1 are
model output, mask=0 are tool/environment output injected into the context.

    python3 tools/trajectories.py runs/<run-id> --model Qwen/Qwen3-30B-A3B-Thinking-2507 [--limit 3] [--session <substr>]

Needs `transformers` (+ tokenizer files; HF_HOME or a cached model). Runs
without numpy (pure-python .npy reader) so it works in the driver image or
any Python with transformers installed — e.g. on the Ray head pod with
HF_HOME=/data/hf-cache HF_HUB_OFFLINE=1 (verified there, 2026-09-30).

"decoded model turns" counts contiguous runs of mask=1 tokens, so it can be
lower than trajectory.json's num_turns: consecutive model turns with no
environment tokens between them are indistinguishable in the token stream
and merge (e.g. 6 decoded vs num_turns=8 on a SWE-bench session).
"""
import argparse, array, io, json, struct, sys, tarfile, zipfile
from pathlib import Path

def read_npz(data: bytes) -> dict:
    out = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            b = z.read(name)
            hlen = struct.unpack("<H", b[8:10])[0]
            hdr = eval(b[10:10 + hlen].decode("latin1"))  # numpy header dict literal
            code = {"<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d", "|i1": "b", "|u1": "B"}[hdr["descr"]]
            arr = array.array(code); arr.frombytes(b[10 + hlen:])
            out[name[:-4]] = list(arr)
    return out

def sessions(rf: Path):
    with tarfile.open(rf / "logs" / "agent-logs.tar.gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        for name, m in members.items():
            if name.endswith("/trajectory.npz"):
                d = name.rsplit("/", 1)[0]
                meta = json.load(tar.extractfile(members[d + "/trajectory.json"])) if d + "/trajectory.json" in members else {}
                yield d, meta, read_npz(tar.extractfile(m).read())

def render(tok, arrays: dict, idx: int = 0) -> str:
    p = arrays[f"traj{idx}_prompt_ids"]; r = arrays[f"traj{idx}_response_ids"]; m = arrays[f"traj{idx}_response_mask"]
    parts = [f"=== PROMPT ({len(p)} tokens) ===\n{tok.decode(p)}\n"]
    turn, cur, cur_kind = 0, [], None
    for t, k in zip(r, m):
        kind = "MODEL" if k else "ENV"
        if kind != cur_kind and cur:
            if cur_kind == "MODEL": turn += 1
            parts.append(f"--- {cur_kind}{' turn ' + str(turn) if cur_kind == 'MODEL' else ''} ({len(cur)} tok) ---\n{tok.decode(cur)}\n")
            cur = []
        cur.append(t); cur_kind = kind
    if cur:
        if cur_kind == "MODEL": turn += 1
        parts.append(f"--- {cur_kind}{' turn ' + str(turn) if cur_kind == 'MODEL' else ''} ({len(cur)} tok) ---\n{tok.decode(cur)}\n")
    return "".join(parts), turn

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_folder"); ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=1); ap.add_argument("--session", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    n = 0
    for d, meta, arrays in sessions(Path(a.run_folder)):
        if a.session and a.session not in d: continue
        text, turns = render(tok, arrays)
        hdr = f"##### {d}\nmeta: {json.dumps(meta.get('trajectories', [{}])[0])}\ndecoded model turns: {turns}\n"
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / (d.replace("/", "_") + ".txt")).write_text(hdr + text)
        else:
            print(hdr + text[:6000] + ("\n...[truncated]" if len(text) > 6000 else ""))
        n += 1
        if n >= a.limit: break
    print(f"rendered {n} session(s)", file=sys.stderr)

if __name__ == "__main__":
    main()
