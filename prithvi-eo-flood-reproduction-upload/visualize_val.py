import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path
from datetime import datetime
import json
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import SequentialSampler
from lightning.pytorch.callbacks import Callback
from terratorch.cli_tools import build_lightning_cli

torch.set_num_threads(4)

import argparse

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Visualize the first three validation samples."
)
parser.add_argument(
    "--config",
    default="configs/sen1floods11_300m.yaml",
    help="Configuration path, relative to the repository root or absolute.",
)
parser.add_argument(
    "--checkpoint",
    required=True,
    help="Fine-tuned Lightning checkpoint path.",
)
parser.add_argument(
    "--output",
    default=None,
    help="Output directory. Defaults to a timestamped visualization folder.",
)
args = parser.parse_args()

# Resolve relative data and weight paths from the repository root.
os.chdir(ROOT)
CONFIG = ROOT / args.config
CHECKPOINT = ROOT / args.checkpoint
OUT = ROOT / (
    args.output or
    ("visualizations/val_first3_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
)

assert CHECKPOINT.is_file(), f"检查点不存在：{CHECKPOINT}"
OUT.mkdir(parents=True)

cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
cfg.pop("ckpt_path", None)
cfg["data"]["init_args"]["batch_size"] = 1
cfg["data"]["init_args"]["num_workers"] = 0
cfg["model"]["init_args"]["plot_on_val"] = 0
cfg["trainer"].update(
    logger=False,
    callbacks=[],
    limit_val_batches=3,
    num_sanity_val_steps=0,
    enable_progress_bar=False,
    default_root_dir=str(OUT),
)

# run=False：只构建对象，不自动启动训练或验证。
cli = build_lightning_cli(args=cfg, run=False)

# 捕获原验证步骤已经算出的预测，不额外改变推理流程。
captured = {}
def capture_output(module, inputs, output):
    captured["prediction"] = (
        output.output.argmax(dim=1).detach().cpu().numpy()
    )

records = []

class SaveVisualizations(Callback):
    def on_validation_start(self, trainer, pl_module):
        loader = trainer.val_dataloaders
        if isinstance(loader, (list, tuple)):
            loader = loader[0]
        assert isinstance(loader.sampler, SequentialSampler), (
            "验证集不是顺序读取，停止以免图片错配"
        )
        self.hook = pl_module.register_forward_hook(capture_output)
        print("验证集影像总数：", len(trainer.datamodule.val_dataset))

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx,
        dataloader_idx=0
    ):
        dataset = trainer.datamodule.val_dataset

        # Dataset 返回缩放后、DataModule 标准化前的影像。
        sample = dataset[batch_idx]
        image = sample["image"].detach().cpu()
        if image.ndim == 4 and image.shape[1] == 1:
            image = image[:, 0]
        assert image.ndim == 3, f"意外影像尺寸：{image.shape}"

        target = batch["mask"][0].detach().cpu().numpy()
        prediction = captured.pop("prediction")[0]
        assert np.array_equal(
            sample["mask"].cpu().numpy(), target
        ), "显示影像与验证标签不对应"
        assert prediction.shape == target.shape

        valid = target != -1
        bands = list(dataset.bands)
        rgb_indices = [bands.index(b) for b in ("RED", "GREEN", "BLUE")]
        rgb = image[rgb_indices].permute(1, 2, 0).numpy()

        # 每个通道做2%至98%拉伸，仅用于显示，不影响模型输入。
        display = np.zeros_like(rgb)
        for c in range(3):
            channel = rgb[..., c]
            usable = valid & np.isfinite(channel)
            if usable.any():
                lo, hi = np.percentile(channel[usable], [2, 98])
                display[..., c] = np.clip(
                    (channel - lo) / max(float(hi - lo), 1e-6),
                    0, 1,
                )
        display = np.nan_to_num(display)

        # 掩膜图：0忽略，1非水体，2水体。
        label_view = np.where(valid, target + 1, 0)
        pred_view = np.where(valid, prediction + 1, 0)
        mask_colors = ["#999999", "#202020", "#168BFF"]

        # 错误图：正确非水体、正确水体、误检、漏检、忽略。
        errors = np.zeros(target.shape, dtype=np.uint8)
        errors[valid & (target == 1) & (prediction == 1)] = 1
        errors[valid & (target == 0) & (prediction == 1)] = 2
        errors[valid & (target == 1) & (prediction == 0)] = 3
        errors[~valid] = 4
        error_colors = [
            "#202020", "#168BFF", "#E53935", "#FFB300", "#999999"
        ]

        intersection = np.sum(valid & (target == 1) & (prediction == 1))
        union = np.sum(valid & ((target == 1) | (prediction == 1)))
        water_iou = float(intersection / union) if union else None

        source = Path(dataset.image_files[batch_idx])
        stem = f"{batch_idx:02d}_{source.stem}"
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        axes[0].imshow(display)
        axes[1].imshow(
            label_view, cmap=ListedColormap(mask_colors),
            vmin=0, vmax=2, interpolation="nearest"
        )
        axes[2].imshow(
            pred_view, cmap=ListedColormap(mask_colors),
            vmin=0, vmax=2, interpolation="nearest"
        )
        axes[3].imshow(
            errors, cmap=ListedColormap(error_colors),
            vmin=0, vmax=4, interpolation="nearest"
        )
        for ax, title in zip(
            axes, ["RGB (display stretch)", "Ground truth",
                   "Prediction", "Error map"]
        ):
            ax.set_title(title)
            ax.axis("off")

        legend_names = [
            "Correct land", "Correct water",
            "False positive", "False negative", "Ignored"
        ]
        fig.legend(
            handles=[
                Patch(color=c, label=n)
                for c, n in zip(error_colors, legend_names)
            ],
            loc="lower center", ncol=5
        )
        score = f"{water_iou:.2%}" if water_iou is not None else "N/A"
        fig.suptitle(f"{source.name} | Water IoU: {score}")
        fig.tight_layout(rect=(0, 0.10, 1, 0.93))
        fig.savefig(OUT / f"{stem}.png", dpi=160)
        plt.close(fig)

        np.savez_compressed(
            OUT / f"{stem}.npz",
            target=target,
            prediction=prediction
        )
        records.append({
            "index": batch_idx,
            "image": str(source),
            "label": str(dataset.segmentation_mask_files[batch_idx]),
            "water_iou": water_iou,
            "figure": f"{stem}.png",
        })
        print("已保存：", OUT / f"{stem}.png", flush=True)

    def on_validation_end(self, trainer, pl_module):
        self.hook.remove()

cli.trainer.callbacks.append(SaveVisualizations())
cli.trainer.validate(
    model=cli.model,
    datamodule=cli.datamodule,
    ckpt_path=str(CHECKPOINT),
)

(OUT / "visualization_record.json").write_text(
    json.dumps({
        "config": str(CONFIG),
        "checkpoint": str(CHECKPOINT),
        "split": "val",
        "selection": "first 3 samples in dataset order",
        "samples": records,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("\n可视化完成，输出目录：", OUT)
