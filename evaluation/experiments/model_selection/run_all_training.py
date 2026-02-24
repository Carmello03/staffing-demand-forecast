import os
import subprocess
from pathlib import Path
from datetime import datetime

# Run order - baselines > native > automl > evaluation
SCRIPTS = [
    Path("evaluation/experiments/model_selection/baseline_metrics.py"),
    Path("evaluation/experiments/model_selection/train_linear_regression.py"),
    Path("evaluation/experiments/model_selection/train_lightgbm.py"),
    Path("evaluation/experiments/model_selection/train_xgboost.py"),
    Path("evaluation/experiments/model_selection/train_flaml.py"),
    Path("evaluation/experiments/model_selection/train_autogluon.py"),
    Path("evaluation/experiments/model_selection/train_pycaret.py"),
]

LOG_DIR = Path("evaluation/experiments/model_selection/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_script(path: Path) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{ts}_{path.stem}.log"

    print(f"\nRunning: {path}")
    print(f"Log:     {log_path}")

    with log_path.open("w", encoding="utf-8", buffering=1) as f:
        f.write(f"Script: {path}\n")
        f.write(f"Start:  {datetime.now().isoformat()}\n\n")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        p = subprocess.Popen(
            ["python", "-u", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in iter(p.stdout.readline, ""):
            print(line, end="")
            f.write(line)

        p.stdout.close()
        rc = p.wait()

        f.write(f"\nEnd:    {datetime.now().isoformat()}\n")
        f.write(f"Exit:   {rc}\n")

    print(f"Finished {path.stem} (exit {rc})")
    return rc


def main():
    # Check scripts exist before starting
    missing = []
    for s in SCRIPTS:
        if not s.exists():
            missing.append(str(s))

    if missing:
        print("Missing scripts:")
        for m in missing:
            print(" -", m)
        return

    # Run them in order
    for s in SCRIPTS:
        rc = run_script(s)
        if rc != 0:
            print(f"\nStopped because {s} failed. Check logs.")
            break


if __name__ == "__main__":
    main()