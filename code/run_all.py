# -*- coding: utf-8 -*-
"""Master runner for the Scientific Reports Revision 2 pipeline.

Chain:
  02b  ISO-year descriptive tables S1/S2 (fast data validation)
  04   core model training with boundary-corrected split
  08   leaky comparator + replay reruns, extra predictions
  08b  post-hoc tables (1/2/3/4/5) + QC figures
  10   reviewer-requested supplementary analyses (S3-S10)
  11   regenerate manuscript base figures (fixed seed/smoothing/scaling)
  12   composite final manuscript figures 2-4
  14   Revision 2: reconciled Table 5 (seed-averaged paired MAE differences)
  15   Revision 2: Figure 2 layout fix (legends outside plot areas)

Run from anywhere: paths are resolved relative to this file.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, '..', 'logs')
os.makedirs(LOGDIR, exist_ok=True)
COMBINED = os.path.join(LOGDIR, 'pipeline_log.txt')

STEPS = [
    ('02b_descriptive_stats_iso.py', 'S1/S2 ISO-year tables'),
    ('04_train_models.py', 'core training (boundary-corrected)'),
    ('08_final_extras.py', 'leaky + replay reruns'),
    ('08b_posthoc_fixed.py', 'post-hoc tables + QC figures'),
    ('10_revision_extras.py', 'reviewer supplementary analyses'),
    ('11_generate_figures_rev.py', 'manuscript base figures'),
    ('12_combine_figures_rev.py', 'final composite figures'),
    ('14_r2_table5_reconcile.py', 'R2: reconciled Table 5 (matches Table 1)'),
    ('15_r2_figure2a_fix.py', 'R2: Figure 2 layout fix'),
]


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(COMBINED, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    log(f'Pipeline start. Python: {sys.executable}')
    t_all = time.time()
    for script, desc in STEPS:
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            log(f'ERROR: missing script {script}')
            sys.exit(1)
        step_log = os.path.join(LOGDIR, script.replace('.py', '.log'))
        log(f'>>> STEP {script} ({desc}) ...')
        t0 = time.time()
        with open(step_log, 'w', encoding='utf-8') as lf:
            p = subprocess.run(
                [sys.executable, '-u', path],
                cwd=HERE, stdout=lf, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
            )
        dt = time.time() - t0
        if p.returncode != 0:
            log(f'ERROR: {script} failed (exit {p.returncode}) after {dt:.0f}s. '
                f'Last 30 lines of {step_log}:')
            with open(step_log, encoding='utf-8', errors='replace') as f:
                tail = f.readlines()[-30:]
            for ln in tail:
                log('    | ' + ln.rstrip())
            sys.exit(1)
        log(f'<<< STEP {script} OK ({dt:.0f}s)')
    log(f'Pipeline COMPLETE in {(time.time() - t_all) / 60:.1f} min.')


if __name__ == '__main__':
    main()
