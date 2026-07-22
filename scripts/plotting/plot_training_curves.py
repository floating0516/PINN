import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"
LOGS_DIR = ROOT / "outputs" / "logs"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STF_RATE_LOSS_NAMES = {"stf_rate", "stf-rate", "stf_rate_wave", "waveform_rate"}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_log() -> Path:
    log_files = list(LOGS_DIR.glob("training_log_*.csv"))
    if not log_files:
        raise FileNotFoundError(f"未找到训练日志文件：{LOGS_DIR}")
    return max(log_files, key=lambda p: p.stat().st_mtime)


def load_log(csv_path: Path) -> dict[str, np.ndarray]:
    epochs, train_loss, train_data, train_phys, val_loss, val_mae, lr = [], [], [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            train_loss.append(float(row["Train_Loss"]))
            train_data.append(float(row["Train_Data_Loss"]))
            train_phys.append(float(row["Train_Phys_Loss"]))
            val_loss.append(float(row["Val_Loss"]))
            val_mae.append(float(row["Val_MAE"]))
            lr.append(float(row["LR"]))
    return {
        "epochs": np.asarray(epochs, dtype=np.int64),
        "train_loss": np.asarray(train_loss, dtype=np.float64),
        "train_data_loss": np.asarray(train_data, dtype=np.float64),
        "train_phys_loss": np.asarray(train_phys, dtype=np.float64),
        "val_loss": np.asarray(val_loss, dtype=np.float64),
        "val_mae": np.asarray(val_mae, dtype=np.float64),
        "lr": np.asarray(lr, dtype=np.float64),
    }


def resolve_rate_representation(config: dict) -> str:
    train_cfg = (config.get("training", {}) or {})
    ds_cfg = (config.get("dataset", {}) or {})
    rate_representation = str(train_cfg.get("rate_representation", "auto")).lower()
    if rate_representation == "auto":
        return "log1p" if "stf_m_ref" in ds_cfg else "linear"
    return rate_representation


def build_loss_summary(config: dict) -> str:
    train_cfg = (config.get("training", {}) or {})
    loss_name = str(train_cfg.get("loss_name", "physics")).lower()
    rate_representation = resolve_rate_representation(config)

    parts = [
        f"loss={loss_name}",
        f"rate={rate_representation}",
    ]

    if loss_name in STF_RATE_LOSS_NAMES:
        stf_cfg = (train_cfg.get("stf_rate_loss", {}) or {})
        ordered_items = [
            ("λ_MSE", stf_cfg.get("lambda_MSE", 1.0)),
            ("λ_phys", stf_cfg.get("lambda_mag", 0.1)),
            ("λ_shape", stf_cfg.get("lambda_shape", 0.1)),
        ]
        if int(config.get("pipeline_version", 1)) != 2:
            ordered_items.insert(
                1,
                ("λ_nonneg", stf_cfg.get("lambda_nonneg", 1.0)),
            )
    else:
        ordered_items = [
            ("w_phys", train_cfg.get("physics_loss_weight", 0.0)),
            ("w_stf", train_cfg.get("stf_loss_weight", 0.0)),
            ("w_smooth", train_cfg.get("stf_smooth_loss_weight", 0.0)),
        ]

    for key, value in ordered_items:
        parts.append(f"{key}={float(value):.4g}")

    return " | ".join(parts)


def add_metric_plot(ax, x: np.ndarray, y: np.ndarray, title: str, ylabel: str, color: str, linestyle: str = "-") -> None:
    ax.plot(x, y, color=color, linewidth=1.8, linestyle=linestyle)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)


def main() -> None:
    config = load_config()
    latest_log = find_latest_log()
    data = load_log(latest_log)
    epochs = data["epochs"]
    loss_summary = build_loss_summary(config)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Training Curves (latest log: {latest_log.name})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    add_metric_plot(axes[0, 0], epochs, data["train_loss"], "Train Total Loss", "Loss", "#0072B2")
    add_metric_plot(axes[0, 1], epochs, data["train_data_loss"], "Train Data Loss", "Loss", "#009E73")
    add_metric_plot(axes[0, 2], epochs, data["train_phys_loss"], "Train Physics Loss", "Loss", "#D55E00")
    add_metric_plot(axes[1, 0], epochs, data["val_loss"], "Validation Loss", "Loss", "#CC79A7")
    add_metric_plot(axes[1, 1], epochs, data["val_mae"], "Validation MAE", "MAE", "#E69F00")
    add_metric_plot(axes[1, 2], epochs, data["lr"], "Learning Rate", "LR", "#000000")
    axes[1, 2].ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))

    best_idx = int(np.nanargmin(data["val_loss"])) if data["val_loss"].size > 0 else None
    if best_idx is not None:
        best_epoch = int(epochs[best_idx])
        best_val_loss = float(data["val_loss"][best_idx])
        axes[1, 0].scatter([best_epoch], [best_val_loss], color="#CC79A7", s=28, zorder=3)
        axes[1, 0].text(
            best_epoch,
            best_val_loss,
            f"  best@{best_epoch}\n  {best_val_loss:.4f}",
            fontsize=8,
            va="bottom",
            ha="left",
        )

    fig.text(
        0.5,
        0.02,
        f"Loss config: {loss_summary}",
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    plt.tight_layout(rect=[0.03, 0.16, 0.98, 0.95])
    save_path = FIGURES_DIR / "latest_training_curves.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"使用最新日志: {latest_log}")
    print(f"已保存图像: {save_path}")
    print(f"损失函数配置: {loss_summary}")


if __name__ == "__main__":
    main()
