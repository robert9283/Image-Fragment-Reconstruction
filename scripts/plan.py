#!/usr/bin/env python3
"""
Experiment run planner.

Commands:
  python scripts/plan.py init            -- populate DB with all planned runs
  python scripts/plan.py list            -- show run table with status
  python scripts/plan.py run-all         -- execute pending runs; Ctrl-C stops after current
  python scripts/plan.py mark-done NAME  -- manually mark a run as done
  python scripts/plan.py reset NAME      -- reset a run back to pending
  python scripts/plan.py add NAME --lambda-adj X --lambda-same Y --beta Z
                                          [--seed N] [--group G] [--notes TEXT]
"""
import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import yaml

SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
DB_PATH      = os.path.join(SCRIPTS_DIR, 'run_plan.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name     TEXT    UNIQUE NOT NULL,
    lambda_adj   REAL    NOT NULL,
    lambda_same  REAL    NOT NULL,
    beta         REAL    NOT NULL,
    seed         INTEGER,
    group_name   TEXT    DEFAULT '',
    status       TEXT    NOT NULL DEFAULT 'pending',
    notes        TEXT    DEFAULT ''
);
"""

# Unified 5-condition ANOVA (5 seeds each).
# Condition (a) adj_only_seed_0..4 is already done and excluded from this list.
PLANNED_RUNS = [
    # (b) same-only — 5 seeds
    *[dict(run_name=f'same_only_seed_{s}',    lambda_adj=0.0, lambda_same=1.0, beta=0.01923, seed=s, group_name='cond_b_same_only',     notes='same-only (cond b)')                for s in range(5)],
    # (c) multi-task lambda_same=1.0 — 5 seeds
    *[dict(run_name=f'multitask_10_seed_{s}', lambda_adj=1.0, lambda_same=1.0, beta=0.01923, seed=s, group_name='cond_c_multitask_10',   notes='multi-task lambda_same=1.0 (cond c)') for s in range(5)],
    # (d) multi-task lambda_same=0.2 — 5 seeds
    *[dict(run_name=f'multitask_02_seed_{s}', lambda_adj=1.0, lambda_same=0.2, beta=0.01923, seed=s, group_name='cond_d_multitask_02',   notes='multi-task lambda_same=0.2 (cond d)') for s in range(5)],
    # (e) adj-only, textbook class-balanced beta=1.0 — 5 seeds
    *[dict(run_name=f'adj_beta1_seed_{s}',    lambda_adj=1.0, lambda_same=0.0, beta=1.0,     seed=s, group_name='cond_e_adj_beta1',      notes='adj-only beta=1 (cond e)')           for s in range(5)],
]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_init(args):
    conn = get_conn()
    inserted = 0
    for r in PLANNED_RUNS:
        try:
            conn.execute(
                "INSERT INTO runs (run_name, lambda_adj, lambda_same, beta, seed, group_name, notes) "
                "VALUES (:run_name, :lambda_adj, :lambda_same, :beta, :seed, :group_name, :notes)", r)
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # already exists, skip
    conn.commit()
    print(f"Initialised: {inserted} new runs added to {DB_PATH}")
    cmd_list(args)


def cmd_list(args):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    if not rows:
        print("No runs in plan. Run: python scripts/plan.py init")
        return
    hdr = f"{'#':<4} {'run_name':<22} {'group':<18} {'l_adj':>6} {'l_same':>7} {'beta':>7} {'seed':>5}  status"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        seed_str = str(r['seed']) if r['seed'] is not None else '-'
        print(f"{r['id']:<4} {r['run_name']:<22} {r['group_name']:<18} "
              f"{r['lambda_adj']:>6.1f} {r['lambda_same']:>7.1f} {r['beta']:>7.5f} "
              f"{seed_str:>5}  {r['status']}")
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print()
    print("  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def cmd_mark_done(args):
    conn = get_conn()
    conn.execute("UPDATE runs SET status='done' WHERE run_name=?", (args.name,))
    conn.commit()
    print(f"Marked done: {args.name}")


def cmd_reset(args):
    conn = get_conn()
    conn.execute("UPDATE runs SET status='pending' WHERE run_name=?", (args.name,))
    conn.commit()
    print(f"Reset to pending: {args.name}")


def cmd_add(args):
    conn = get_conn()
    conn.execute(
        "INSERT INTO runs (run_name, lambda_adj, lambda_same, beta, seed, group_name, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (args.name, args.lambda_adj, args.lambda_same, args.beta,
         args.seed, args.group or '', args.notes or ''))
    conn.commit()
    print(f"Added: {args.name}")


def _clean_run(run_name):
    """Remove run dir and results.jsonl entry for a run before rerunning."""
    run_dir = os.path.join(PROJECT_ROOT, 'runs', run_name)
    if os.path.isdir(run_dir):
        import shutil
        shutil.rmtree(run_dir)
    results_path = os.path.join(PROJECT_ROOT, 'results.jsonl')
    if os.path.exists(results_path):
        with open(results_path) as f:
            lines = [l for l in f if json.loads(l).get('run') != run_name]
        with open(results_path, 'w') as f:
            f.writelines(lines)


def _write_config(row):
    cfg_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name']            = row['run_name']
    cfg['model']               = 'fragment-adjacency-predictor'
    cfg['n_images']            = 10
    cfg['max_iterations']      = 25000
    cfg['eval_every']          = 250
    cfg['patience']            = 10
    cfg['beta']                = float(row['beta'])
    cfg['lambda_adj']          = float(row['lambda_adj'])
    cfg['lambda_same']         = float(row['lambda_same'])
    cfg['n_eval_batches']      = 20
    cfg['balanced_clustering'] = True
    cfg['notes']               = row['notes']
    if row['seed'] is not None:
        cfg['seed'] = int(row['seed'])
    else:
        cfg.pop('seed', None)
    cfg.pop('pos_weight_same', None)
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def cmd_run_all(args):
    conn = get_conn()
    stop_after_current = [False]

    def on_sigint(sig, frame):
        print("\n[plan] Ctrl-C received — will stop after the current run finishes.")
        stop_after_current[0] = True

    signal.signal(signal.SIGINT, on_sigint)

    while True:
        row = conn.execute(
            "SELECT * FROM runs WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            print("[plan] All pending runs complete.")
            _run_analysis()
            break

        run_name = row['run_name']
        print(f"\n{'='*60}")
        print(f"[plan] Starting: {run_name}  (lambda_adj={row['lambda_adj']}  "
              f"lambda_same={row['lambda_same']}  seed={row['seed']})")
        print(f"{'='*60}")

        _clean_run(run_name)
        _write_config(row)

        conn.execute("UPDATE runs SET status='running' WHERE run_name=?", (run_name,))
        conn.commit()

        log_path = os.path.join(PROJECT_ROOT, 'runs', f'{run_name}.log')
        os.makedirs(os.path.join(PROJECT_ROOT, 'runs'), exist_ok=True)
        with open(log_path, 'w') as log_f:
            proc = subprocess.Popen(
                [sys.executable, os.path.join(PROJECT_ROOT, 'main.py')],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
            proc.wait()

        if proc.returncode == 0:
            conn.execute("UPDATE runs SET status='done' WHERE run_name=?", (run_name,))
            print(f"[plan] Done: {run_name}")
        else:
            conn.execute("UPDATE runs SET status='pending' WHERE run_name=?", (run_name,))
            print(f"[plan] FAILED (exit {proc.returncode}): {run_name} — left as pending")
        conn.commit()

        if stop_after_current[0]:
            print("[plan] Stopping as requested.")
            break


def _run_analysis():
    print("\n[plan] Running analysis scripts...")
    for script in ['scripts/compare_runs.py', 'scripts/anova_analysis.py']:
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, script)], cwd=PROJECT_ROOT)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('init',      help='Populate DB with planned runs')
    sub.add_parser('list',      help='Show run table')
    sub.add_parser('run-all',   help='Execute pending runs (Ctrl-C stops cleanly)')

    p_done = sub.add_parser('mark-done', help='Mark a run as done')
    p_done.add_argument('name')

    p_reset = sub.add_parser('reset', help='Reset a run to pending')
    p_reset.add_argument('name')

    p_add = sub.add_parser('add', help='Add a run to the plan')
    p_add.add_argument('name')
    p_add.add_argument('--lambda-adj',  type=float, required=True)
    p_add.add_argument('--lambda-same', type=float, required=True)
    p_add.add_argument('--beta',        type=float, required=True)
    p_add.add_argument('--seed',        type=int,   default=None)
    p_add.add_argument('--group',       default='')
    p_add.add_argument('--notes',       default='')

    args = p.parse_args()
    {'init': cmd_init, 'list': cmd_list, 'run-all': cmd_run_all,
     'mark-done': cmd_mark_done, 'reset': cmd_reset, 'add': cmd_add}[args.cmd](args)


if __name__ == '__main__':
    main()
