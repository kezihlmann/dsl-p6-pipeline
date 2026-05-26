from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BASE_MASKS_ROOT = Path("eval_mask")
GROUND_TRUTH_SUFFIX = "_mask_ground_truth.png"
REPORT_PATH = Path("mask_quality_report_eval_mask.txt")

# Add methods here as new mask files become available next to each ground truth mask.
EVAL_METHODS = ["sam3", "colors", "grounded_sam_vit_b"]

METRIC_COLUMNS = [
    "IoU",
    "Dice",
    "Precision",
    "Recall",
    "F1",
    "AP_Foreground",
    "AP_Background",
    "mAP",
]

METHOD_LABELS = {
    "sam3": "SAM 3",
    "colors": "Basic Color",
    "grounded_sam_vit_b": "GroundedSAM vit_b",
}


def load_mask(mask_path: Path):
    """Load mask image and convert to binary values (0 or 1)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


def calculate_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def calculate_dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_true, mask_pred).sum()
    total = mask_true.sum() + mask_pred.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AP for a binary class with binary scores (0/1) efficiently."""
    y_true = y_true.astype(np.uint8).ravel()
    y_score = (y_score > 0).astype(np.uint8).ravel()

    positives = int(y_true.sum())
    total = int(y_true.size)
    negatives = total - positives
    if positives == 0:
        return 1.0

    tp_high = int(np.sum((y_true == 1) & (y_score == 1)))
    fp_high = int(np.sum((y_true == 0) & (y_score == 1)))

    recall_high = tp_high / positives
    precision_high = tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else 1.0
    precision_low = positives / total if total > 0 else 1.0

    # For binary scores, the PR curve has only two recall jumps: at score=1 and score=0.
    precision_high_envelope = max(precision_high, precision_low)
    ap = (recall_high * precision_high_envelope) + ((1.0 - recall_high) * precision_low)
    return float(ap)


def calculate_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int(np.sum((prediction == 1) & (ground_truth == 1)))
    fp = int(np.sum((prediction == 1) & (ground_truth == 0)))
    fn = int(np.sum((prediction == 0) & (ground_truth == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ap_foreground = average_precision_binary(ground_truth, prediction)
    ap_background = average_precision_binary(1 - ground_truth, 1 - prediction)
    mean_ap = 0.5 * (ap_foreground + ap_background)

    return {
        "IoU": calculate_iou(ground_truth, prediction),
        "Dice": calculate_dice(ground_truth, prediction),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1_score),
        "AP_Foreground": float(ap_foreground),
        "AP_Background": float(ap_background),
        "mAP": float(mean_ap),
    }


def method_label(method_key: str) -> str:
    if method_key in METHOD_LABELS:
        return METHOD_LABELS[method_key]
    return method_key.replace("_", " ").title()


def extract_image_id_from_gt(gt_path: Path) -> str:
    return gt_path.name[: -len(GROUND_TRUTH_SUFFIX)]


def discover_ground_truth_masks(base_dir: Path) -> list:
    return sorted(base_dir.rglob(f"*{GROUND_TRUTH_SUFFIX}"))


def build_method_mask_path(gt_path: Path, method_key: str) -> Path:
    image_id = extract_image_id_from_gt(gt_path)
    return gt_path.parent / f"{image_id}_mask_{method_key}.png"


def evaluate_method(
    method_key: str,
    gt_records: list,
):
    rows = []
    missing = 0
    load_fail = 0
    shape_mismatch = 0

    for record in gt_records:
        image_id = record["image_id"]
        gt_mask = record["gt_mask"]
        mask_path = build_method_mask_path(record["gt_path"], method_key)

        if not mask_path.exists():
            missing += 1
            continue

        pred_mask = load_mask(mask_path)
        if pred_mask is None:
            load_fail += 1
            continue

        if gt_mask.shape != pred_mask.shape:
            shape_mismatch += 1
            continue

        metrics = calculate_metrics(gt_mask, pred_mask)
        row = {
            "Image": image_id,
            "MethodKey": method_key,
            "Method": method_label(method_key),
        }
        row.update(metrics)
        rows.append(row)

    stats = {
        "evaluated": len(rows),
        "missing": missing,
        "load_fail": load_fail,
        "shape_mismatch": shape_mismatch,
    }
    return rows, stats


def build_metric_table(mean_or_std_df: pd.DataFrame) -> pd.DataFrame:
    table = mean_or_std_df[METRIC_COLUMNS].transpose().reset_index()
    table = table.rename(columns={"index": "Metric"})
    return table


def write_report(df: pd.DataFrame, method_stats: dict, total_ground_truth: int) -> Path:
    mean_df = df.groupby("Method")[METRIC_COLUMNS].mean().sort_index()
    std_df = df.groupby("Method")[METRIC_COLUMNS].std().fillna(0.0).sort_index()

    summary_table = build_metric_table(mean_df)
    std_table = build_metric_table(std_df)

    lines = []
    lines.append("=" * 120)
    lines.append("MASK QUALITY EVALUATION REPORT")
    lines.append("Dataset: eval_mask")
    lines.append(f"Ground truth masks found: {total_ground_truth}")
    lines.append(f"Images evaluated: {df['Image'].nunique()}")
    lines.append("=" * 120)
    lines.append("")

    lines.append("[METHOD AVAILABILITY]")
    for method_key in sorted(method_stats.keys(), key=method_label):
        stats = method_stats[method_key]
        lines.append(
            f"- {method_label(method_key)}: evaluated={stats['evaluated']}, "
            f"missing={stats['missing']}, load_fail={stats['load_fail']}, shape_mismatch={stats['shape_mismatch']}"
        )
    lines.append("")

    lines.append("[SUMMARY STATISTICS - MEAN]")
    lines.append(summary_table.to_string(index=False))
    lines.append("")

    lines.append("[STANDARD DEVIATIONS]")
    lines.append(std_table.to_string(index=False))
    lines.append("")

    lines.append("[KEY FINDINGS]")
    for metric in METRIC_COLUMNS:
        best_method = mean_df[metric].idxmax()
        best_score = mean_df[metric].max()
        lines.append(f"[+] {metric}: {best_method} achieves best score ({best_score:.4f})")
    lines.append("")

    lines.append("[METRIC DEFINITIONS]")
    lines.append("- IoU: Intersection over Union")
    lines.append("- Dice: 2 * Intersection / (Sum of masks)")
    lines.append("- Precision: TP / (TP + FP)")
    lines.append("- Recall: TP / (TP + FN)")
    lines.append("- F1: Harmonic mean of Precision and Recall")
    lines.append("- AP_Foreground: Average Precision for foreground class")
    lines.append("- AP_Background: Average Precision for background class")
    lines.append("- mAP: (AP_Foreground + AP_Background) / 2")
    lines.append("")

    lines.append("[PER-IMAGE RESULTS]")
    lines.append(df.to_string(index=False))

    report_path = REPORT_PATH
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def process_dataset(base_dir: Path) -> bool:
    gt_paths = discover_ground_truth_masks(base_dir)
    if not gt_paths:
        print(f"[WARN] No ground truth masks found under: {base_dir}")
        return False

    gt_records = []
    for gt_path in gt_paths:
        image_id = extract_image_id_from_gt(gt_path)
        gt_mask = load_mask(gt_path)
        if gt_mask is None:
            print(f"[WARN] Could not load ground truth: {gt_path.name}")
            continue
        gt_records.append(
            {
                "image_id": image_id,
                "gt_path": gt_path,
                "gt_mask": gt_mask,
            }
        )

    if not gt_records:
        print("[WARN] No valid ground truth masks could be loaded")
        return False

    methods = EVAL_METHODS

    all_rows = []
    method_stats = {}

    print("\n[INFO] Processing eval_mask dataset")
    print(f"[INFO] Ground truth images: {len(gt_records)}")
    print(f"[INFO] Evaluating methods: {', '.join(method_label(k) for k in methods)}")

    for method_key in methods:
        rows, stats = evaluate_method(method_key, gt_records)
        method_stats[method_key] = stats
        all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No valid method predictions were evaluated")
        return False

    df = pd.DataFrame(all_rows)
    report_path = write_report(df, method_stats, len(gt_records))
    print(f"[OK] Report saved: {report_path}")
    return True


def main():
    if not BASE_MASKS_ROOT.exists():
        print(f"[ERROR] Base folder does not exist: {BASE_MASKS_ROOT}")
        return

    if not process_dataset(BASE_MASKS_ROOT):
        print("[ERROR] Evaluation failed")
        return

    print("\n[DONE] Evaluation finished")


if __name__ == "__main__":
    main()
