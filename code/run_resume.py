# -*- coding: utf-8 -*-
"""Resume runner: continue the pipeline from step 08 after the key fix
(02b and 04 already completed successfully at 09:47-10:04)."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, '..', 'logs')
os.makedirs(LOGDIR, exist_ok=True)
COMBINED = os.path.join(LOGDIR, 'pipeline_log.txt')

STEPS = [
    ('08_final_extras.py', 'leaky + replay reruns (key fix)'),
    ('08b_posthoc_fixed.py', 'post-hoc tables + QC figures'),
    ('10_revision_extras.py', 'reviewer supplementary analyses'),
    ('11_generate_figures_rev.py', 'manuscript base figures'),
    ('12_combine_figures_rev.py', 'final composite figures'),
]


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(COMBINED, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    log(f'Resume pipeline start (from 08). Python: {sys.executable}')
    t_all = time.time()
    for script, desc in STEPS:
        path = os.path.join(HERE, script)
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
    log(f'Resume pipeline COMPLETE in {(time.time() - t_all) / 60:.1f} min.')


if __name__ == '__main__':
    main()
