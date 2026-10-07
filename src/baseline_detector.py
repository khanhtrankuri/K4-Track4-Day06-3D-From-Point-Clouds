"""Benchmark a pretrained MMDetection3D KITTI point-cloud detector.

The model is never trained here. Config and checkpoint must be provided by the user.
API reference: https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/get_started.md
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starter.datasets import list_frames, load_frame
from starter.projection import box3d_corners_cam


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def prediction_arrays(sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = sample.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    return boxes, scores, labels


def gt_bev_corners(frame) -> list[np.ndarray]:
    inverse = np.linalg.inv(frame["calib"].T_cam_velo)
    result = []
    for obj in frame["labels"]:
        if obj.type != "Car":
            continue
        corners = box3d_corners_cam(obj)[:4]
        hom = np.column_stack((corners, np.ones(4)))
        result.append((hom @ inverse.T)[:, :2])
    return result


def pred_bev_corners(box: np.ndarray) -> np.ndarray:
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                      [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def centers_matched(gt: list[np.ndarray], boxes: np.ndarray, max_distance: float) -> list[bool]:
    if len(boxes) == 0:
        return [False] * len(gt)
    centers = np.asarray([b[:2] for b in boxes])
    gt_centers = np.asarray([np.mean(g, axis=0) for g in gt])
    distances = np.linalg.norm(gt_centers[:, None, :] - centers[None, :, :], axis=2)
    matched = [False] * len(gt)
    while distances.size:
        gt_idx, pred_idx = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[gt_idx, pred_idx] > max_distance:
            break
        matched[gt_idx] = True
        distances[gt_idx, :] = np.inf
        distances[:, pred_idx] = np.inf
    return matched


def draw_bev(frame, boxes: np.ndarray, scores: np.ndarray, out: Path,
             max_distance: float, title: str) -> None:
    points = frame["points"]
    valid = np.isfinite(points[:, :2]).all(axis=1)
    points = points[valid][::8]
    gt = gt_bev_corners(frame)
    matched = centers_matched(gt, boxes, max_distance)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(points[:, 0], points[:, 1], s=0.15, c="0.55", alpha=0.45, rasterized=True)
    for poly, hit in zip(gt, matched):
        p = np.vstack((poly, poly[0]))
        ax.plot(p[:, 0], p[:, 1], c="limegreen" if hit else "red", lw=2.0)
    for box, score in zip(boxes, scores):
        p = pred_bev_corners(box)
        p = np.vstack((p, p[0]))
        ax.plot(p[:, 0], p[:, 1], c="deepskyblue", lw=1.3)
        ax.text(box[0], box[1], f"{score:.2f}", fontsize=6, color="blue")
    ax.set(xlim=(0, 75), ylim=(-40, 40), xlabel="x LiDAR (m)", ylabel="y LiDAR (m)", title=title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="MMDetection3D KITTI config")
    ap.add_argument("--checkpoint", required=True, type=Path, help="pretrained KITTI checkpoint")
    ap.add_argument("--data-root", type=Path, default=Path("data/kitti_mini"))
    ap.add_argument("--frames", nargs="+", default=["000001", "000004", "000008", "000011", "000049"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=20, help="timed runs after one warm-up")
    ap.add_argument("--max-center-distance", type=float, default=2.0)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    for p in (args.config, args.checkpoint):
        if not p.is_file():
            ap.error(f"missing file: {p}")
    if args.repeats < 20:
        ap.error("--repeats must be at least 20")
    if not set(args.frames).issubset(list_frames(args.data_root)):
        ap.error("one or more --frames are absent from --data-root")
    np.random.seed(42)
    try:
        import torch
        from mmdet3d.apis import init_model, inference_detector
    except ImportError as exc:
        ap.error(f"MMDetection3D is required: {exc}")
    torch.manual_seed(42)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(42)
    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    model.eval()
    classes = tuple(model.dataset_meta.get("classes", ()))
    if classes and classes[0].lower() != "car":
        ap.error(f"This benchmark expects KITTI Car at label 0; model classes: {classes}")
    cfg = model.cfg
    meta = {"config": str(args.config), "checkpoint": str(args.checkpoint),
            "classes": classes, "device": args.device, "seed": 42,
            "frames": args.frames, "thresholds": args.thresholds,
            "max_center_distance_m": args.max_center_distance,
            "repeats": args.repeats,
            "point_cloud_range": cfg.get("point_cloud_range"),
            "voxel_size": cfg.get("voxel_size"),
            "model_test_cfg": str(cfg.model.get("test_cfg", ""))}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baseline_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    rows = []
    figure_dir = args.out_dir / "figures"
    first_fail = False
    all_scores = []
    for i, frame_id in enumerate(args.frames):
        frame = load_frame(args.data_root, frame_id)
        cloud = args.data_root / "training" / "velodyne" / f"{frame_id}.bin"
        result, _ = inference_detector(model, str(cloud))
        boxes, scores, labels = prediction_arrays(result)
        car = labels == 0
        boxes, scores = boxes[car], scores[car]
        all_scores.extend(scores.tolist())
        gt = gt_bev_corners(frame)
        for threshold in args.thresholds:
            keep = scores >= threshold
            kept_boxes, kept_scores = boxes[keep], scores[keep]
            hits = centers_matched(gt, kept_boxes, args.max_center_distance)
            row = {"frame_id": frame_id, "score_thr": threshold, "gt_cars": len(gt),
                   "pred_cars": len(kept_boxes), "matched_gt_cars": sum(hits),
                   "center_recall": sum(hits) / len(gt) if gt else float("nan"),
                   "mean_score": float(kept_scores.mean()) if len(kept_scores) else float("nan"),
                   "min_range_m": float(np.linalg.norm(kept_boxes[:, :2], axis=1).min()) if len(kept_boxes) else float("nan"),
                   "max_range_m": float(np.linalg.norm(kept_boxes[:, :2], axis=1).max()) if len(kept_boxes) else float("nan")}
            rows.append(row)
            if threshold == args.thresholds[0]:
                frame_out = figure_dir / f"demo_baseline_bev_{frame_id}.png"
                draw_bev(frame, kept_boxes, kept_scores, frame_out,
                         args.max_center_distance, f"{frame_id}: PointPillars, score >= {threshold}")
                if i == 0:
                    draw_bev(frame, kept_boxes, kept_scores, figure_dir / "demo_baseline_bev.png",
                             args.max_center_distance, f"{frame_id}: PointPillars, score >= {threshold}")
            if (not first_fail and threshold == args.thresholds[0]
                    and any(not hit for hit in hits)):
                draw_bev(frame, kept_boxes, kept_scores, figure_dir / "fail_01_missed_car.png",
                         args.max_center_distance,
                         f"{frame_id}, score >= {threshold}: red GT has no nearby prediction")
                first_fail = True
    columns = list(rows[0])
    with (args.out_dir / "baseline_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    thresholds = np.asarray(args.thresholds)
    total_predictions = [sum(r["pred_cars"] for r in rows if r["score_thr"] == t) for t in thresholds]
    total_matches = [sum(r["matched_gt_cars"] for r in rows if r["score_thr"] == t) for t in thresholds]
    total_gt = sum(r["gt_cars"] for r in rows if r["score_thr"] == thresholds[0])
    fig, left = plt.subplots(figsize=(7, 4))
    left.plot(thresholds, total_predictions, "o-", color="tab:blue", label="Predicted Car boxes")
    left.set(xlabel="Score threshold", ylabel="Predicted Car boxes")
    right = left.twinx()
    right.plot(thresholds, np.asarray(total_matches) / total_gt, "s--", color="tab:red",
               label="GT center recall")
    right.set(ylabel="GT center recall", ylim=(0, 1.05))
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="center right")
    left.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_threshold_tradeoff.png", dpi=150)
    plt.close(fig)
    if not first_fail:
        print("No missed Car found for selected frames/thresholds; no failure figure was fabricated.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_scores, bins=np.linspace(0, 1, 21), color="steelblue", edgecolor="white")
    ax.set(xlabel="Car confidence score", ylabel="Number of boxes", title="Scores before thresholding, five KITTI frames")
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_score_histogram.png", dpi=150)
    plt.close(fig)
    cloud = args.data_root / "training" / "velodyne" / f"{args.frames[0]}.bin"
    inference_detector(model, str(cloud))  # warm-up; excluded
    synchronize(args.device)
    times_ms = []
    for _ in range(args.repeats):
        synchronize(args.device)
        start = time.perf_counter()
        inference_detector(model, str(cloud))
        synchronize(args.device)
        times_ms.append((time.perf_counter() - start) * 1000)
    with (args.out_dir / "baseline_latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "latency_ms"])
        writer.writerows((i + 1, t) for i, t in enumerate(times_ms))
    print(f"frames={len(args.frames)} rows={len(rows)} latency p50={np.percentile(times_ms, 50):.1f} ms "
          f"p95={np.percentile(times_ms, 95):.1f} ms")


if __name__ == "__main__":
    main()
