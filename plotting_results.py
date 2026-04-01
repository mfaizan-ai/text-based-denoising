import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RUN_LABELS = {
    "baseline_full": "Our model",
    "combined_training": "Baseline",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected metrics file: {path}")
    return pd.read_csv(path)


def style_plot() -> None:
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["legend.edgecolor"] = "#333333"
    plt.rcParams["axes.titlepad"] = 10
    plt.rcParams["axes.labelpad"] = 5


def build_run_list(runs_path: Path, run_names: list[str] | None = None) -> list[Path]:
    if run_names:
        return [runs_path / run_name for run_name in run_names if (runs_path / run_name).exists()]
    return [runs_path / run_name for run_name in ["baseline_full", "combined_training"] if (runs_path / run_name).exists()]


def ensure_clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_train_metric(metric: str, train_data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, ax = plt.subplots()
    palette = sns.color_palette("tab10")

    for idx, (run_name, train_df) in enumerate(train_data.items()):
        label = RUN_LABELS.get(run_name, run_name)
        if metric not in train_df.columns:
            continue
        ax.plot(train_df["step"], train_df[metric], label=label, color=palette[idx], linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"Train {metric.replace('_', ' ').upper()} Comparison")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"train_{metric}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_val_metric(metric: str, val_data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, ax = plt.subplots()
    palette = sns.color_palette("tab10")

    for idx, (run_name, val_df) in enumerate(val_data.items()):
        label = RUN_LABELS.get(run_name, run_name)
        if metric not in val_df.columns:
            continue
        ax.plot(val_df["step"], val_df[metric], label=label, color=palette[idx], linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"Validation {metric.replace('_', ' ').upper()} Comparison")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"val_{metric}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_test_metric(metric: str, test_data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    tasks = sorted({task for df in test_data.values() for task in df["task"].tolist()})
    x = range(len(tasks))
    width = 0.35
    palette = sns.color_palette("tab10")

    fig, ax = plt.subplots()
    for idx, (run_name, df) in enumerate(test_data.items()):
        label = RUN_LABELS.get(run_name, run_name)
        values = [float(df.loc[df["task"] == task, metric].item()) if task in df["task"].values else float("nan") for task in tasks]
        positions = [xi + (idx - 0.5) * width for xi in x]
        ax.bar(positions, values, width=width, label=label, color=palette[idx], edgecolor="black", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.set_xlabel("Task")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Test {metric.upper()} Comparison by Task")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"test_{metric}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training, validation, and test metrics for WindowSeat runs.")
    parser.add_argument("--runs_path", type=Path, required=True, help="Path to the runs directory")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory where plot images will be saved")
    parser.add_argument(
        "--run_names",
        nargs="*",
        default=None,
        help="Optional run directories to plot; defaults to baseline_full and combined_training if available",
    )
    args = parser.parse_args()

    runs_path = args.runs_path
    if not runs_path.exists() or not runs_path.is_dir():
        raise FileNotFoundError(f"Runs path does not exist: {runs_path}")

    output_dir = args.output_dir or runs_path / "plots"
    ensure_clean_output(output_dir)
    style_plot()

    selected_runs = build_run_list(runs_path, args.run_names)
    if not selected_runs:
        raise ValueError("No run directories found to plot. Check --runs_path and optionally use --run_names.")

    train_data: dict[str, pd.DataFrame] = {}
    val_data: dict[str, pd.DataFrame] = {}
    test_data: dict[str, pd.DataFrame] = {}

    for run_path in selected_runs:
        run_name = run_path.name
        print(f"Processing {run_name}")
        train_data[run_name] = load_csv(run_path / "train_metrics.csv")
        val_data[run_name] = load_csv(run_path / "val_metrics.csv")
        test_csv = run_path / "test_metrics.csv"
        if test_csv.exists():
            test_data[run_name] = load_csv(test_csv)
        else:
            print(f"Skipping test metrics for {run_name}: missing {test_csv}")

    plot_train_metric("loss_psnr", train_data, output_dir)
    plot_train_metric("loss_ssim", train_data, output_dir)
    plot_train_metric("train_psnr", train_data, output_dir)
    plot_train_metric("train_ssim", train_data, output_dir)

    plot_val_metric("val_psnr", val_data, output_dir)
    plot_val_metric("val_ssim", val_data, output_dir)

    if test_data:
        plot_test_metric("psnr", test_data, output_dir)
        plot_test_metric("ssim", test_data, output_dir)

    print(f"Plots saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
