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
    """
    Open (or create) the SQLite plan database and return a connection.

    Creates the 'runs' table if it does not exist yet. The connection uses
    sqlite3.Row as its row factory so columns are accessible by name.

    Returns:
        sqlite3.Connection: Open database connection with SCHEMA applied.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_init(args):
    """
    Populate the plan database with all entries in PLANNED_RUNS.

    Skips runs that already exist (by unique run_name). Prints a summary of
    how many new rows were inserted, then calls cmd_list to show the full
    table.

    Args:
        args: Parsed argparse namespace (not used; present for dispatch
            uniformity).
    """
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
    """
    Print all runs in the plan database, one per row.

    Columns shown: id, run_name, group, lambda_adj, lambda_same, beta, seed,
    status. A count summary (pending / running / done) is printed below the
    table.

    Args:
        args: Parsed argparse namespace (not used; present for dispatch
            uniformity).
    """
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
    """
    Manually mark a run as 'done' in the plan database.

    Useful when a run was completed outside of cmd_run_all (e.g. run manually
    or on a different machine) and needs to be recorded as finished.

    Args:
        args: Parsed argparse namespace; must have attribute 'name' (run_name).
    """
    conn = get_conn()
    conn.execute("UPDATE runs SET status='done' WHERE run_name=?", (args.name,))
    conn.commit()
    print(f"Marked done: {args.name}")


def cmd_reset(args):
    """
    Reset a run's status to 'pending' so it will be re-executed by cmd_run_all.

    Args:
        args: Parsed argparse namespace; must have attribute 'name' (run_name).
    """
    conn = get_conn()
    conn.execute("UPDATE runs SET status='pending' WHERE run_name=?", (args.name,))
    conn.commit()
    print(f"Reset to pending: {args.name}")


def cmd_add(args):
    """
    Insert a new run into the plan database with the given hyperparameters.

    Args:
        args: Parsed argparse namespace; must have attributes 'name',
            'lambda_adj', 'lambda_same', 'beta', 'seed', 'group', 'notes'.
    """
    conn = get_conn()
    conn.execute(
        "INSERT INTO runs (run_name, lambda_adj, lambda_same, beta, seed, group_name, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (args.name, args.lambda_adj, args.lambda_same, args.beta,
         args.seed, args.group or '', args.notes or ''))
    conn.commit()
    print(f"Added: {args.name}")


def _clean_run(run_name):
    """
    Delete a run's directory and remove its entry from results.jsonl.

    Called before rerunning a failed or reset run to ensure a clean slate.
    Silently skips if the run directory or results.jsonl does not exist.

    Args:
        run_name (str): The run identifier matching a subdirectory of runs/.
    """
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
    """
    Overwrite config.yaml with the hyperparameters from a plan database row.

    Reads the current config.yaml, updates only the keys that are controlled
    by the experiment plan (run_name, model, n_images, max_iterations,
    eval_every, patience, beta, lambda_adj, lambda_same, n_eval_batches,
    balanced_clustering, notes, seed), and writes the result back.

    Args:
        row (sqlite3.Row): A row from the plan database with all required
            hyperparameter columns.
    """
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
    """
    Execute all pending runs in order, streaming output to stdout and a log file.

    For each pending run: cleans any prior artifacts, writes config.yaml, marks
    the run as 'running', spawns main.py as a subprocess, and marks it 'done'
    on success or resets it to 'pending' on failure. Pressing Ctrl-C sets a
    flag that stops the loop cleanly after the current run finishes. When all
    runs complete, calls _run_analysis() to regenerate summary tables.

    Args:
        args: Parsed argparse namespace (not used; present for dispatch
            uniformity).
    """
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
    """
    Run the post-experiment analysis scripts after all runs finish.

    Sequentially executes compare_runs.py and anova_analysis.py from the
    project root so that summary tables and figures are up to date.
    """
    print("\n[plan] Running analysis scripts...")
    for script in ['scripts/compare_runs.py', 'scripts/anova_analysis.py']:
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, script)], cwd=PROJECT_ROOT)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """
    Parse CLI arguments and dispatch to the appropriate command function.

    Supported subcommands: init, list, run-all, mark-done, reset, add.
    Each subcommand maps to a cmd_* function defined above.
    """
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
