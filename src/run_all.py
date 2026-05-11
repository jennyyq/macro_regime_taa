import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "download_data.py",
    "preprocess_data.py",
    "detect_regimes.py",
    "backtest_naive_strategy.py",
    "backtest_bl_strategy.py",
    "backtest_ridge_strategy.py",
    "plot_figure10.py",
]


def run_script(script_name: str):
    script_path = PROJECT_ROOT / "src" / script_name

    print("\n" + "=" * 80)
    print(f"RUNNING: {script_name}")
    print("=" * 80)

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

    print(f"\nFINISHED: {script_name}")


def main():
    print("=" * 80)
    print("MACRO REGIME TAA PIPELINE STARTED")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")

    for script in SCRIPTS:
        run_script(script)

    print("\n" + "=" * 80)
    print("ALL SCRIPTS FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()