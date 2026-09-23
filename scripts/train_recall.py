"""isoFLOPs and isoParams comparison of the GRT against a non-recurrent stack.

The paper's two regimes, at a scale that fits on a laptop:

* isoFLOPs -- the GRT with ``n_pre + n_core * R + n_coda`` block executions is
  compared against a plain stack with that many *unique* blocks, so both do the
  same number of layer applications and the plain stack stores far more
  parameters.
* isoParams -- the GRT's unique block count ``n_pre + n_core + n_coda`` is
  matched by a plain stack with the same number of unique blocks, so both store
  similar parameters and the GRT does R times more layer applications.

Both models are trained on the same synthetic multi-query associative recall
batch stream (same seed, same order) and evaluated on held-out batches.

    python scripts/train_recall.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grt.blocks import counting_block_factory  # noqa: E402
from grt.config import GRTConfig  # noqa: E402
from grt.model import GRT, PlainStack  # noqa: E402
from grt.tasks import MQARConfig, make_batch, to_mlx  # noqa: E402
from grt.train import evaluate, train  # noqa: E402

PAPER = {
    "title": "Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation",
    "arxiv": "arXiv:2608.15062",
    "authors": "Amr Hegazy, Amr Alanwar, Mostafa Elhoushi",
}

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def measured_block_applications(model, task: MQARConfig, R: int, counter: list) -> int:
    """Count real block executions in one noise-free forward pass."""
    tokens, _, _ = to_mlx(make_batch(np.random.default_rng(0), task))
    counter[0] = 0
    model(tokens, R=R)
    mx.eval()
    return counter[0]


def build_grt(cfg: GRTConfig, counter: list) -> GRT:
    model = GRT(cfg, block_factory=counting_block_factory(counter))
    mx.eval(model.parameters())
    return model


def build_plain(cfg: GRTConfig, n_layers: int, counter: list) -> PlainStack:
    model = PlainStack(cfg, n_layers=n_layers, block_factory=counting_block_factory(counter))
    mx.eval(model.parameters())
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=8, help="key/value pairs per sequence")
    ap.add_argument("--keys", type=int, default=16, help="distinct keys (chance = 1/keys)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--core-depths", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--recurrences", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR))
    args = ap.parse_args()

    task = MQARConfig(n_pairs=args.pairs, n_keys=args.keys, batch_size=args.batch)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "isoflops_isoparams.txt"
    json_path = out_dir / "isoflops_isoparams.json"

    log_file = open(log_path, "w")

    def log(message: str = "") -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    log(f"{PAPER['title']}")
    log(f"{PAPER['arxiv']} -- {PAPER['authors']}")
    log("Toy-scale isoFLOPs / isoParams comparison (not a reproduction of the paper).")
    log()
    log("Task: multi-query associative recall, sequence = k v pairs then the keys")
    log(f"      again in a shuffled order. pairs={task.n_pairs} keys={task.n_keys}"
        f" seq_len={task.seq_len} vocab={task.vocab_size}")
    log(f"      chance accuracy = {task.chance_accuracy:.4f} over the whole vocabulary"
        f" ({task.chance_accuracy_over_values:.4f} if a model emitted only values);")
    log(f"      scored on the {task.n_pairs} query positions only")
    log("Model: dim=%d heads=%d mlp_ratio=4, learned positions, pre-norm blocks"
        % (args.dim, args.heads))
    log(f"Train: AdamW lr={args.lr} cosine with {args.warmup}-step warmup,"
        f" weight_decay=0.01, grad clip 1.0, batch={args.batch},"
        f" {args.steps} steps, state noise sigma_x=0.1, gate noise sigma_g=0.1,")
    log(f"       depth sampling off (fixed R), one seed ({args.seed}) shared by every"
        " run so all models see the same batch stream")
    log(f"Eval:  {args.eval_batches} held-out batches, noise-free, at the full R")
    log()

    base_cfg = dict(
        vocab_size=task.vocab_size,
        max_seq_len=task.seq_len,
        dim=args.dim,
        n_heads=args.heads,
    )

    runs = []          # every trained model
    run_by_tag = {}

    def train_one(tag, kind, model, R, extra):
        log(f"[{tag}] {kind}: unique_blocks={model.unique_blocks}"
            f" params={model.parameter_count()} layer_applications="
            f"{model.layer_applications(R)}")
        t0 = time.time()
        history = train(
            model,
            task,
            R=R,
            steps=args.steps,
            peak_lr=args.lr,
            warmup=args.warmup,
            seed=args.seed,
            log_every=max(250, args.steps // 8),
            eval_every=max(250, args.steps // 4),
            eval_batches=max(4, args.eval_batches // 4),
            log=log,
        )
        metrics = evaluate(model, task, R=R, n_batches=args.eval_batches, seed=args.seed + 7)
        record = dict(
            tag=tag,
            kind=kind,
            R=R,
            unique_blocks=model.unique_blocks,
            parameters=model.parameter_count(),
            layer_applications=model.layer_applications(R),
            held_out_loss=metrics["loss"],
            held_out_accuracy=metrics["accuracy"],
            train_loss_final=history["train_loss"][-1],
            wall_time_s=history["wall_time_s"],
        )
        record.update(extra)
        runs.append(record)
        run_by_tag[tag] = record
        log(f"    -> held-out loss {metrics['loss']:.4f}  accuracy {metrics['accuracy']:.4f}"
            f"  ({history['wall_time_s']:.1f}s)")
        return record

    # ---------------------------------------------------------------- GRT runs
    counter = [0]
    for core in args.core_depths:
        for R in args.recurrences:
            tag = f"grt-c{core}-R{R}"
            cfg = GRTConfig(
                **base_cfg, n_prelude=1, n_core=core, n_coda=1, recurrence=R
            )
            mx.random.seed(args.seed)
            model = build_grt(cfg, counter)
            record = train_one(tag, "grt", model, R, {"core_depth": core})
            record["measured_block_applications"] = measured_block_applications(
                model, task, R, counter
            )
            if record["measured_block_applications"] != record["layer_applications"]:
                raise RuntimeError(
                    f"measured {record['measured_block_applications']} block applications "
                    f"but the formula says {record['layer_applications']}"
                )

    # ------------------------------------------------- non-recurrent baselines
    # isoFLOPs needs a stack with n_pre + n_core * R + n_coda unique blocks for
    # every GRT configuration; isoParams needs one with n_pre + n_core + n_coda.
    needed = set()
    isoflops_pairs = []
    for core in args.core_depths:
        for R in args.recurrences:
            n_isoflops = 2 + core * R
            needed.add(n_isoflops)
            isoflops_pairs.append((f"grt-c{core}-R{R}", n_isoflops, core, R))
    isoparams_pairs = []
    for core in args.core_depths:
        n_isoparams = 2 + core
        needed.add(n_isoparams)
        for R in args.recurrences:
            isoparams_pairs.append((f"grt-c{core}-R{R}", n_isoparams, core, R))

    for n_layers in sorted(needed):
        tag = f"plain-{n_layers}L"
        mx.random.seed(args.seed)
        model = build_plain(GRTConfig(**base_cfg), n_layers, counter)
        record = train_one(tag, "plain", model, n_layers, {})
        record["measured_block_applications"] = measured_block_applications(
            model, task, n_layers, counter
        )
        if record["measured_block_applications"] != n_layers:
            raise RuntimeError("plain stack block application count is wrong")

    if len(runs) != len(run_by_tag):
        raise RuntimeError("duplicate run tags")

    # ------------------------------------------------------------------ tables
    def pair_record(grt_tag, plain_tag, regime, note):
        g, p = run_by_tag[grt_tag], run_by_tag[plain_tag]
        return dict(
            regime=regime,
            grt=grt_tag,
            plain=plain_tag,
            note=note,
            layer_applications_grt=g["layer_applications"],
            layer_applications_plain=p["layer_applications"],
            parameters_grt=g["parameters"],
            parameters_plain=p["parameters"],
            parameter_ratio_plain_over_grt=p["parameters"] / g["parameters"],
            loss_grt=g["held_out_loss"],
            loss_plain=p["held_out_loss"],
            loss_delta=g["held_out_loss"] - p["held_out_loss"],
            accuracy_grt=g["held_out_accuracy"],
            accuracy_plain=p["held_out_accuracy"],
        )

    isoflops_table = [
        pair_record(grt_tag, f"plain-{n}L", "isoFLOPs",
                    "same block applications, different unique parameter count")
        for grt_tag, n, _, _ in isoflops_pairs
    ]
    isoparams_table = [
        dict(pair_record(grt_tag, f"plain-{n}L", "isoParams",
                         "same unique block count; the GRT applies its core R times"),
             R=R, core_depth=core)
        for grt_tag, n, core, R in isoparams_pairs
    ]

    def table(title, rows):
        log()
        log(title)
        log("-" * len(title))
        header = (f"{'GRT':<14}{'plain':<10}{'apps':>6}{'params G':>10}{'params P':>10}"
                  f"{'P/G':>6}{'loss G':>9}{'loss P':>9}{'dLoss':>8}{'acc G':>8}{'acc P':>8}")
        log(header)
        for r in rows:
            log(f"{r['grt']:<14}{r['plain']:<10}{r['layer_applications_grt']:>6}"
                f"{r['parameters_grt']:>10}{r['parameters_plain']:>10}"
                f"{r['parameter_ratio_plain_over_grt']:>6.2f}"
                f"{r['loss_grt']:>9.4f}{r['loss_plain']:>9.4f}{r['loss_delta']:>+8.4f}"
                f"{r['accuracy_grt']:>8.4f}{r['accuracy_plain']:>8.4f}")
        wins = sum(1 for r in rows if r["loss_delta"] < 0)
        log(f"GRT lower held-out loss in {wins}/{len(rows)} cells"
            f" (mean delta {sum(r['loss_delta'] for r in rows) / len(rows):+.4f})")

    log()
    log("=" * 100)
    log("Summary (held-out loss and query accuracy; dLoss < 0 means the GRT is better)")
    log("=" * 100)
    log()
    log("All trained models")
    log("-----------------")
    log(f"{'run':<16}{'kind':<7}{'R':>4}{'blocks':>8}{'params':>9}{'apps':>6}"
        f"{'measured':>10}{'loss':>9}{'acc':>8}{'sec':>7}")
    for r in runs:
        log(f"{r['tag']:<16}{r['kind']:<7}{r['R']:>4}{r['unique_blocks']:>8}"
            f"{r['parameters']:>9}{r['layer_applications']:>6}"
            f"{r.get('measured_block_applications', 0):>10}{r['held_out_loss']:>9.4f}"
            f"{r['held_out_accuracy']:>8.4f}{r['wall_time_s']:>7.1f}")

    table("isoFLOPs: GRT vs a plain stack with the same number of block applications",
          isoflops_table)
    table("isoParams: GRT vs a plain stack with the same number of unique blocks",
          isoparams_table)

    log()
    log("Reading these numbers")
    log("---------------------")
    log("Every number above comes from this script's own run. This is a toy scale:"
        " dim=%d," % args.dim)
    log(f"{task.seq_len}-token sequences, {args.steps} steps, one seed. The paper's"
        " comparisons are")
    log("GPT-2 Small/Medium/Large scale (124M-774M parameters, ~9.8B training tokens)")
    log("and nothing here is comparable to them.")

    payload = dict(
        paper=PAPER,
        script="scripts/train_recall.py",
        disclaimer=(
            "Toy-scale independent implementation. Numbers are not comparable to the "
            "paper's GPT-2-scale results."
        ),
        task=dict(
            name="multi-query associative recall",
            n_pairs=task.n_pairs,
            n_keys=task.n_keys,
            seq_len=task.seq_len,
            vocab_size=task.vocab_size,
            batch_size=task.batch_size,
            chance_accuracy=task.chance_accuracy,
        ),
        model=dict(dim=args.dim, n_heads=args.heads, mlp_ratio=4,
                   n_prelude=1, n_coda=1),
        training=dict(steps=args.steps, peak_lr=args.lr, warmup=args.warmup,
                      weight_decay=0.01, grad_clip=1.0, seed=args.seed,
                      depth_sampling=False, state_noise_std=0.1, gate_noise_std=0.1),
        evaluation=dict(batches=args.eval_batches, noise="off"),
        runs=runs,
        isoflops=isoflops_table,
        isoparams=isoparams_table,
        sweeps=dict(core_depths=args.core_depths, recurrences=args.recurrences),
    )
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    log()
    log(f"wrote {json_path}")
    log(f"wrote {log_path}")
    log_file.close()

    print()
    print("Done. Tables above were written to results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
