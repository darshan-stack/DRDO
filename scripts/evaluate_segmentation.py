#!/usr/bin/env python3
"""Formal SemanticPOSS evaluation for the trained FFEM range-image model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ffem.io.semantic_poss import SemanticPOSSDataset, CLASS_NAMES
from ffem.perception.factory import build_segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument(
        "--checkpoint",
        default="models/checkpoints/semanticposs_range_model.pt",
    )
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--plots-dir",
        default="outputs/semanticposs_eval_plots",
    )
    ap.add_argument(
        "--output",
        default="outputs/semanticposs_eval.json",
    )
    args = ap.parse_args()

    ds = SemanticPOSSDataset(args.data_root, args.sequences)
    seg, selected = build_segmenter(
        "torch_range",
        args.checkpoint,
        7,
        device=args.device,
    )

    k = len(CLASS_NAMES)
    cm = np.zeros((k, k), dtype=np.int64)
    range_bins = [0.0, 10.0, 30.0, 60.0, 100.0]
    by_range = {
        f"{int(range_bins[i])}_{int(range_bins[i + 1])}m": [0, 0]
        for i in range(len(range_bins) - 1)
    }

    processed = 0
    total_points = 0
    limit = len(ds) if args.max_scans <= 0 else min(len(ds), args.max_scans)

    for i in range(limit):
        points, intensity, truth, _ = ds[i]
        pred, _ = seg.predict(points, intensity)
        truth = np.asarray(truth)
        pred = np.asarray(pred)
        valid = (
            (truth >= 0)
            & (truth < k)
            & (pred >= 0)
            & (pred < k)
        )
        np.add.at(cm, (truth[valid], pred[valid]), 1)
        total_points += int(valid.sum())

        ranges = np.linalg.norm(points, axis=1)
        for j in range(len(range_bins) - 1):
            mask = (
                (ranges >= range_bins[j])
                & (ranges < range_bins[j + 1])
                & valid
            )
            count = int(mask.sum())
            correct = int(np.sum(pred[mask] == truth[mask])) if count else 0
            key = f"{int(range_bins[j])}_{int(range_bins[j + 1])}m"
            by_range[key][0] += count
            by_range[key][1] += correct

        processed += 1
        if (i + 1) % 10 == 0 or i + 1 == limit:
            print(f"evaluated {i + 1}/{limit}")

    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1).astype(float)
    iou_den = tp + fp + fn
    p_den = tp + fp
    r_den = tp + fn
    iou = np.divide(tp, iou_den, out=np.zeros(k), where=iou_den > 0)
    precision = np.divide(tp, p_den, out=np.zeros(k), where=p_den > 0)
    recall = np.divide(tp, r_den, out=np.zeros(k), where=r_den > 0)
    present = support > 0

    report = {
        "checkpoint": selected,
        "device": getattr(seg, "device", args.device),
        "dataset": "semanticposs",
        "sequences": args.sequences,
        "scans": processed,
        "points": total_points,
        "class_names": list(CLASS_NAMES),
        "confusion_matrix": cm.tolist(),
        "per_class_iou": dict(zip(CLASS_NAMES, iou.tolist())),
        "per_class_precision": dict(zip(CLASS_NAMES, precision.tolist())),
        "per_class_recall": dict(zip(CLASS_NAMES, recall.tolist())),
        "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "overall_accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "accuracy_by_range": {
            key: {
                "points": values[0],
                "accuracy": float(values[1] / max(values[0], 1)),
            }
            for key, values in by_range.items()
        },
    }

    plots = []
    try:
        import matplotlib.pyplot as plt

        plot_dir = Path(args.plots_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(cm, interpolation="nearest")
        fig.colorbar(image, ax=ax)
        ax.set_title("SemanticPOSS Confusion Matrix")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_xticks(range(k), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(k), CLASS_NAMES)
        fig.tight_layout()
        confusion_path = plot_dir / "confusion_matrix.png"
        fig.savefig(confusion_path, dpi=180)
        plt.close(fig)
        plots.append(str(confusion_path))

        labels = list(by_range)
        values = [report["accuracy_by_range"][key]["accuracy"] for key in labels]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Range")
        ax.set_title("SemanticPOSS Accuracy by Range")
        fig.tight_layout()
        range_path = plot_dir / "accuracy_by_range.png"
        fig.savefig(range_path, dpi=180)
        plt.close(fig)
        plots.append(str(range_path))
    except ImportError:
        report["plot_status"] = "matplotlib not installed; plots not generated"
    else:
        report["plots"] = plots

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "mean_iou": report["mean_iou"],
                "macro_precision": report["macro_precision"],
                "macro_recall": report["macro_recall"],
                "overall_accuracy": report["overall_accuracy"],
                "device": report["device"],
            },
            indent=2,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
