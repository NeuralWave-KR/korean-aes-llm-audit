"""
Reproduce all paper numbers end to end.

Runs, in dependency order:
  1. e09_human_ceiling   -> outputs/e9_human_ceiling/    (human ceiling, Tables 1,2)
  2. e10_factorial       -> outputs/e10_factorial/       (factorial design, Tables 3,4)
  3. e7l_original        -> outputs/e7l_original/         (audited original pipeline; 0.625 arm)
  4. e11_honest_eval     -> outputs/e11_honest_eval/      (representative score 0.581, Table 5)
  5. tables_6_7          -> outputs/tables_6_7/           (Tables 6,7; needs 1,3,4)
  6. e9b_grade_variance  -> outputs/e9b_grade_variance/   (§5.4: per-grade score SD vs QWK; needs 5)
  7. e12_pilot_latest_model -> outputs/e12_pilot/         (supplementary: latest-model pilot, §5.4;
                                                           self-skips if pilot ratings absent)

No network / no API calls: every stage reads only the local cached data.
Then run `python verify.py` to check the outputs against the reported values.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
STAGES = ["e09_human_ceiling", "e10_factorial", "e7l_original",
          "e11_honest_eval", "tables_6_7", "e9b_grade_variance",
          "e12_pilot_latest_model"]


def main():
    t0 = time.time()
    for i, stage in enumerate(STAGES, 1):
        script = os.path.join(SRC, f"{stage}.py")
        print(f"\n{'='*60}\n[{i}/{len(STAGES)}] {stage}\n{'='*60}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, script], cwd=HERE)
        if r.returncode != 0:
            print(f"\n!! stage {stage} failed (exit {r.returncode})")
            sys.exit(r.returncode)
        print(f"   done in {time.time()-t:.0f}s")
    print(f"\nAll stages done in {time.time()-t0:.0f}s. "
          f"Now run:  {sys.executable} verify.py")


if __name__ == "__main__":
    main()
