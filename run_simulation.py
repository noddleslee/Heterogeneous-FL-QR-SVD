"""
Orchestration script for Federated Learning simulation.

GPU scheduling
--------------
The script auto-detects available GPUs and assigns clients to GPUs in batches.

  num_clients=4, num_gpus=4  → 1 batch  of 4 (all parallel)
  num_clients=4, num_gpus=2  → 2 batches of 2 (serial batches, parallel within)
  num_clients=6, num_gpus=4  → batch[4] then batch[2]

Aggregation always sees all client adapters at once — batching is transparent.

FL modes
--------
  FedAvg   (default): uniform rank, --rank
  FlexLoRA           : heterogeneous ranks, --flexlora --ranks 4 16 32 64
"""

import argparse
import json
import math
import os
import subprocess

PYTHON_EXE = ["/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python", "-u"]
BASE_DIR   = "/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora"


# ── GPU detection ──────────────────────────────────────────────────────────────

def detect_num_gpus() -> int:
    """Return number of visible CUDA GPUs via nvidia-smi (fallback: 1)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode()
        n = len([l for l in out.strip().splitlines() if l.strip()])
        return max(n, 1)
    except Exception:
        return 1


# ── per-client command builder ─────────────────────────────────────────────────

def _build_client_cmd(
    client_id, num_clients, round_idx, output_dir, base_model,
    local_rank, global_rank, epochs, max_steps, total_rounds,
    global_model_path, flexlora, hetlora=False, het_lambda=0.01, het_gamma=0.5,
):
    cmd = PYTHON_EXE + [
        os.path.join(BASE_DIR, "train_client.py"),
        "--client_id",    str(client_id),
        "--num_clients",  str(num_clients),
        "--round",        str(round_idx),
        "--total_rounds", str(total_rounds),
        "--output_dir",   output_dir,
        "--base_model",   base_model,
        "--rank",         str(local_rank),
        "--global_rank",  str(global_rank),
        "--epochs",       str(epochs),
        "--max_steps",    str(max_steps),
    ]
    if global_model_path:
        cmd += ["--global_model_path", global_model_path]
    if flexlora:
        cmd.append("--flexlora")
    if hetlora:
        cmd += ["--hetlora", "--het_lambda", str(het_lambda), "--het_gamma", str(het_gamma)]
    return cmd


# ── batched client execution ───────────────────────────────────────────────────

def _run_client_batch(batch, gpu_ids, round_idx, output_dir, num_clients,
                      base_model, global_rank, epochs, max_steps, total_rounds,
                      global_model_path, flexlora, hetlora=False, het_lambda=0.01, het_gamma=0.5):
    """
    Launch one batch of clients in parallel, one process per GPU slot.

    batch    : list of (client_id, local_rank) for this batch
    gpu_ids  : GPU IDs to use (len == len(batch))

    Returns True if all processes succeeded, False otherwise.
    """
    processes = []
    for slot, (client_id, local_rank) in enumerate(batch):
        log_path = os.path.join(output_dir, f"round_{round_idx}", f"client_{client_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        cmd = _build_client_cmd(
            client_id=client_id,
            num_clients=num_clients,
            round_idx=round_idx,
            output_dir=output_dir,
            base_model=base_model,
            local_rank=local_rank,
            global_rank=global_rank,
            epochs=epochs,
            max_steps=max_steps,
            total_rounds=total_rounds,
            global_model_path=global_model_path,
            flexlora=flexlora,
            hetlora=hetlora,
            het_lambda=het_lambda,
            het_gamma=het_gamma,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[slot])

        print(f"  Client {client_id} (rank={local_rank}) → GPU {gpu_ids[slot]}")
        with open(log_path, "w") as f:
            p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        processes.append((client_id, p))
        import time
        time.sleep(4) # Stagger launches to prevent unsloth inspect.getsource crash

    ok = True
    for client_id, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"  ERROR: Client {client_id} failed. "
                  f"Check round_{round_idx}/client_{client_id}.log")
            ok = False
        else:
            print(f"  Client {client_id} done.")
    return ok


# ── round orchestration ────────────────────────────────────────────────────────

def run_round(
    round_idx,
    num_clients,
    output_dir,
    base_model,
    client_ranks,     # list[int], length == num_clients
    global_rank,
    epochs,
    max_steps,
    total_rounds,
    gpu_ids,          # list[int], detected or user-specified
    flexlora=False,
    hetlora=False,
    het_lambda=0.01,
    het_gamma=0.5,
):
    num_gpus = len(gpu_ids)
    num_batches = math.ceil(num_clients / num_gpus)

    print(f"  ROUND {round_idx}"
          + (f"  [FlexLoRA ranks={client_ranks}, global_rank={global_rank}]"
             if flexlora else
             f"  [HetLoRA ranks={client_ranks}, global_rank={global_rank}]"
             if hetlora else
             f"  [FedAvg rank={global_rank}]"))
    print(f"  {num_clients} clients / {num_gpus} GPUs → {num_batches} batch(es)")
    print("=" * 60)

    global_model_path = (
        os.path.join(output_dir, f"round_{round_idx - 1}", "global_model")
        if round_idx > 0 else None
    )

    # ── client list: [(client_id, local_rank), …] ──
    all_clients = [(i, client_ranks[i]) for i in range(num_clients)]

    # ── process in GPU-sized batches ──
    for b_idx in range(num_batches):
        batch = all_clients[b_idx * num_gpus : (b_idx + 1) * num_gpus]
        batch_gpu_ids = gpu_ids[: len(batch)]   # may be shorter than gpu_ids on last batch

        if num_batches > 1:
            print(f"\n  -- Batch {b_idx + 1}/{num_batches}: "
                  f"clients {[c for c,_ in batch]} --")

        ok = _run_client_batch(
            batch             = batch,
            gpu_ids           = batch_gpu_ids,
            round_idx         = round_idx,
            output_dir        = output_dir,
            num_clients       = num_clients,
            base_model        = base_model,
            global_rank       = global_rank,
            epochs            = epochs,
            max_steps         = max_steps,
            total_rounds      = total_rounds,
            global_model_path = global_model_path,
            flexlora          = flexlora,
            hetlora           = hetlora,
            het_lambda        = het_lambda,
            het_gamma         = het_gamma,
        )
        if not ok:
            return False

    # ── aggregate all clients (unchanged regardless of batching) ──
    print(f"\nAggregating {num_clients} clients for round {round_idx} …")
    client_dirs = [
        os.path.join(output_dir, f"round_{round_idx}", f"client_{i}")
        for i in range(num_clients)
    ]
    global_dir = os.path.join(output_dir, f"round_{round_idx}", "global_model")

    agg_cmd = PYTHON_EXE + [
        os.path.join(BASE_DIR, "aggregate.py"),
        "--client_dirs", *client_dirs,
        "--output_dir",  global_dir,
        "--mode",        "hetlora" if hetlora else "flexlora" if flexlora else "fedavg",
        "--global_rank", str(global_rank),
    ]
    if global_model_path:
        agg_cmd += ["--global_model_path", global_model_path]
        
    if subprocess.run(agg_cmd).returncode != 0:
        print(f"ERROR: Aggregation failed in round {round_idx}")
        return False

    # ── async SLURM evaluation ──
    _submit_eval(round_idx, output_dir, base_model, global_dir)
    return True


def _submit_eval(round_idx, output_dir, base_model, global_dir):
    print(f"Submitting async evaluation job for round {round_idx} …")
    script = os.path.join(output_dir, f"round_{round_idx}", "submit_eval.sh")
    result_json = os.path.join(output_dir, f"round_{round_idx}", "eval_results.json")

    with open(script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --job-name=eval_rnd{round_idx}
#SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output={os.path.abspath(output_dir)}/round_{round_idx}/eval_%j.out

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

{PYTHON_EXE[0]} {os.path.join(BASE_DIR, "eval_pathvqa.py")} \\
    --base_model {base_model} \\
    --peft_path {os.path.abspath(global_dir)} \\
    --output_json {os.path.abspath(result_json)} \\
    --n_samples 500
""")
    subprocess.run(["sbatch", script])


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FL simulation for PathVQA (adaptive GPU batching)")
    parser.add_argument("--num_rounds",  type=int, default=3)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--output_dir",  type=str,
                        default="/mimer/NOBACKUP/groups/bloom/shenghui/hfl/fed_sim_output")
    parser.add_argument("--base_model",  type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--epochs",      type=int, default=1)
    parser.add_argument("--max_steps",   type=int, default=-1)

    # GPU control
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: auto-detect all visible GPUs)")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="Specific GPU IDs to use, e.g. --gpu_ids 0 1  "
                             "(overrides --num_gpus)")

    # FedAvg
    parser.add_argument("--rank", type=int, default=16,
                        help="Uniform LoRA rank for FedAvg mode")

    # FlexLoRA
    parser.add_argument("--flexlora", action="store_true",
                        help="Enable FlexLoRA with heterogeneous client ranks")
    parser.add_argument("--ranks", type=int, nargs="+", default=None,
                        help="Per-client LoRA ranks for FlexLoRA/HetLoRA (cycles if shorter than "
                             "num_clients). Default: [4, 16, 32, 64]")

    # HetLoRA
    parser.add_argument("--hetlora", action="store_true",
                        help="Enable HetLoRA (EMNLP 2024)")
    parser.add_argument("--het_lambda", type=float, default=0.01,
                        help="Pruning regularization coefficient for HetLoRA")
    parser.add_argument("--het_gamma", type=float, default=0.5,
                        help="Pruning ratio for HetLoRA (fraction of ranks to consider for pruning)")

    args = parser.parse_args()

    # ── resolve GPU list ──
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif args.num_gpus is not None:
        gpu_ids = list(range(args.num_gpus))
    else:
        gpu_ids = list(range(detect_num_gpus()))

    # ── resolve per-client ranks ──
    if args.flexlora:
        base_ranks   = args.ranks or [4, 16, 32, 64]
        client_ranks = [base_ranks[i % len(base_ranks)] for i in range(args.num_clients)]
        global_rank  = max(client_ranks)
        mode_str     = f"FlexLoRA  client_ranks={client_ranks}  global_rank={global_rank}"
    elif args.hetlora:
        base_ranks   = args.ranks or [4, 16, 32, 64]
        client_ranks = [base_ranks[i % len(base_ranks)] for i in range(args.num_clients)]
        global_rank  = max(client_ranks)
        mode_str     = f"HetLoRA  client_ranks={client_ranks}  global_rank={global_rank} lambda={args.het_lambda} gamma={args.het_gamma}"
    else:
        client_ranks = [args.rank] * args.num_clients
        global_rank  = args.rank
        mode_str     = f"FedAvg  rank={global_rank}"

    os.makedirs(args.output_dir, exist_ok=True)

    num_batches_per_round = math.ceil(args.num_clients / len(gpu_ids))
    print(f"\n{'='*60}")
    print(f"  FL SIMULATION  —  {mode_str}")
    print(f"  Rounds   : {args.num_rounds}")
    print(f"  Clients  : {args.num_clients}")
    print(f"  GPUs     : {gpu_ids}  ({len(gpu_ids)} total)")
    print(f"  Batches/round: {num_batches_per_round}")
    print(f"  Output   : {args.output_dir}")
    print(f"{'='*60}")

    for r in range(args.num_rounds):
        # ── [HetLoRA] update client ranks from persistent state ──
        if args.hetlora and r > 0:
            for i in range(args.num_clients):
                rank_file = os.path.join(args.output_dir, f"client_{i}_rank.json")
                if os.path.exists(rank_file):
                    with open(rank_file) as f:
                        rank_info = json.load(f)
                    old_rank = client_ranks[i]
                    client_ranks[i] = rank_info["rank"]
                    if client_ranks[i] != old_rank:
                        print(f"  [HetLoRA] Client {i} rank updated: {old_rank} → {client_ranks[i]} (pruned in round {rank_info['round']})")
            # Keep global_rank in sync so the server model doesn't hold
            # unnecessary capacity after all clients have pruned.
            new_global_rank = max(client_ranks)
            if new_global_rank != global_rank:
                print(f"  [HetLoRA] global_rank updated: {global_rank} → {new_global_rank}")
                global_rank = new_global_rank
            print(f"  [HetLoRA] Current ranks: {client_ranks}, global_rank: {global_rank}")

        ok = run_round(
            round_idx    = r,
            num_clients  = args.num_clients,
            output_dir   = args.output_dir,
            base_model   = args.base_model,
            client_ranks = client_ranks,
            global_rank  = global_rank,
            epochs       = args.epochs,
            max_steps    = args.max_steps,
            total_rounds = args.num_rounds,
            gpu_ids      = gpu_ids,
            flexlora     = args.flexlora,
            hetlora      = args.hetlora,
            het_lambda   = args.het_lambda,
            het_gamma    = args.het_gamma,
        )
        if not ok:
            print(f"\nSimulation aborted at round {r}.")
            break

    print("\n" + "=" * 60)
    print("  FEDERATED SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
