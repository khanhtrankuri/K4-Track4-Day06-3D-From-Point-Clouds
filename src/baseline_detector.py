"""Benchmark a pretrained MMDetection3D KITTI point-cloud detector.

The model is never trained here. Config and checkpoint must be provided by the user.
Outputs are prefixed with --tag so several models can be compared on the same frames.
API reference: https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/get_started.md
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starter.datasets import list_frames, load_frame
from starter.projection import box3d_corners_cam, velo_to_cam

# KITTI "Moderate" difficulty: bbox height >= 25 px, occlusion <= 1, truncation <= 0.3.
MODERATE_MIN_HEIGHT_PX = 25.0
MODERATE_MAX_OCCLUSION = 1
MODERATE_MAX_TRUNCATION = 0.3


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


def car_labels(frame) -> list:
    return [obj for obj in frame["labels"] if obj.type == "Car"]


def is_moderate(obj) -> bool:
    height = obj.bbox[3] - obj.bbox[1]
    return (height >= MODERATE_MIN_HEIGHT_PX and obj.occluded <= MODERATE_MAX_OCCLUSION
            and obj.truncated <= MODERATE_MAX_TRUNCATION)


def gt_bev_corners(frame) -> list[np.ndarray]:
    inverse = np.linalg.inv(frame["calib"].T_cam_velo)
    result = []
    for obj in car_labels(frame):
        corners = box3d_corners_cam(obj)[:4]
        hom = np.column_stack((corners, np.ones(4)))
        result.append((hom @ inverse.T)[:, :2])
    return result


def points_in_gt_boxes(frame) -> list[int]:
    """Number of LiDAR points inside each GT Car box (rectified camera frame)."""
    points = frame["points"][:, :3]
    points = velo_to_cam(points[np.isfinite(points).all(axis=1)], frame["calib"])
    counts = []
    for obj in car_labels(frame):
        h, w, l = obj.dimensions
        c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        local = (points - obj.location) @ rot  # R^T (p - loc), row-vector form
        inside = ((np.abs(local[:, 0]) <= l / 2) & (local[:, 1] <= 0) & (local[:, 1] >= -h)
                  & (np.abs(local[:, 2]) <= w / 2))
        counts.append(int(inside.sum()))
    return counts


def in_camera_fov(frame, boxes: np.ndarray) -> np.ndarray:
    """True if the BEV centre of a LiDAR-frame box projects inside image_2 width."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    centers = boxes[:, :3].copy()
    centers[:, 2] += boxes[:, 5] / 2  # mmdet3d LiDAR boxes use bottom centre
    cam = velo_to_cam(centers, frame["calib"])
    proj = np.column_stack((cam, np.ones(len(cam)))) @ frame["calib"].P2.T
    u = proj[:, 0] / np.where(proj[:, 2] > 0, proj[:, 2], np.nan)
    return (cam[:, 2] > 0.1) & (u >= 0) & (u < frame["image"].shape[1])


def fov_rays(frame, max_depth: float = 80.0) -> list[np.ndarray]:
    """Left/right image borders as BEV rays in LiDAR frame."""
    P2, width = frame["calib"].P2, frame["image"].shape[1]
    inverse = np.linalg.inv(frame["calib"].T_cam_velo)
    rays = []
    for u in (0, width):
        x_cam = (u - P2[0, 2]) / P2[0, 0] * max_depth
        pts = np.array([[0, 0, 0, 1], [x_cam, 0, max_depth, 1]], dtype=float)
        rays.append((pts @ inverse.T)[:, :2])
    return rays


def pred_bev_corners(box: np.ndarray) -> np.ndarray:
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                      [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def match_centers(gt: list[np.ndarray], boxes: np.ndarray,
                  max_distance: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one BEV centre matching. Returns (gt_matched, pred_matched) masks."""
    gt_hit = np.zeros(len(gt), dtype=bool)
    pred_hit = np.zeros(len(boxes), dtype=bool)
    if len(boxes) == 0 or len(gt) == 0:
        return gt_hit, pred_hit
    centers = np.asarray([b[:2] for b in boxes])
    gt_centers = np.asarray([np.mean(g, axis=0) for g in gt])
    distances = np.linalg.norm(gt_centers[:, None, :] - centers[None, :, :], axis=2)
    while True:
        gt_idx, pred_idx = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[gt_idx, pred_idx] > max_distance:
            break
        gt_hit[gt_idx] = pred_hit[pred_idx] = True
        distances[gt_idx, :] = np.inf
        distances[:, pred_idx] = np.inf
    return gt_hit, pred_hit


def draw_bev(frame, boxes: np.ndarray, scores: np.ndarray, out: Path,
             max_distance: float, title: str, annotate_missed: bool = False) -> None:
    points = frame["points"]
    valid = np.isfinite(points[:, :2]).all(axis=1)
    points = points[valid][::8]
    gt = gt_bev_corners(frame)
    objs = car_labels(frame)
    counts = points_in_gt_boxes(frame)
    matched, _ = match_centers(gt, boxes, max_distance)
    fov = in_camera_fov(frame, boxes)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(points[:, 0], points[:, 1], s=0.15, c="0.55", alpha=0.45, rasterized=True)
    for ray in fov_rays(frame):
        ax.plot(ray[:, 0], ray[:, 1], ls=":", c="purple", lw=1.0)
    for poly, hit, obj, n in zip(gt, matched, objs, counts):
        p = np.vstack((poly, poly[0]))
        ax.plot(p[:, 0], p[:, 1], c="limegreen" if hit else "red", lw=2.0)
        if annotate_missed and not hit:
            cx, cy = poly.mean(axis=0)
            ax.text(cx + 1.5, cy, f"occ={obj.occluded}\npts={n}", fontsize=7, color="red")
    for box, score, inside in zip(boxes, scores, fov):
        p = pred_bev_corners(box)
        p = np.vstack((p, p[0]))
        ax.plot(p[:, 0], p[:, 1], c="deepskyblue" if inside else "darkorange", lw=1.3)
        ax.text(box[0], box[1], f"{score:.2f}", fontsize=6, color="blue")
    handles = [plt.Line2D([], [], c="deepskyblue", label="prediction (camera FOV)"),
               plt.Line2D([], [], c="darkorange", label="prediction (outside FOV, no label)"),
               plt.Line2D([], [], c="limegreen", lw=2, label="GT matched"),
               plt.Line2D([], [], c="red", lw=2, label="GT missed"),
               plt.Line2D([], [], c="purple", ls=":", label="camera FOV")]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    ax.set(xlim=(0, 75), ylim=(-40, 40), xlabel="x LiDAR (m)", ylabel="y LiDAR (m)", title=title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def time_inference(model, inputs, device: str, repeats: int) -> list[float]:
    from mmdet3d.apis import inference_detector
    inference_detector(model, inputs)  # warm-up; excluded
    synchronize(device)
    times_ms = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        inference_detector(model, inputs)
        synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)
    return times_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="MMDetection3D KITTI config")
    ap.add_argument("--checkpoint", required=True, type=Path, help="pretrained KITTI checkpoint")
    ap.add_argument("--tag", default="pointpillars", help="prefix for output files, e.g. pointpillars/second")
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
    hardware = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else platform.processor()
    meta = {"tag": args.tag, "config": args.config.as_posix(), "checkpoint": args.checkpoint.as_posix(),
            "classes": classes, "device": args.device, "hardware": hardware,
            "torch": torch.__version__, "seed": 42,
            "frames": args.frames, "thresholds": args.thresholds,
            "max_center_distance_m": args.max_center_distance,
            "repeats": args.repeats,
            "point_cloud_range": cfg.get("point_cloud_range"),
            "voxel_size": cfg.get("voxel_size"),
            "model_test_cfg": str(cfg.model.get("test_cfg", ""))}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    rows, gt_rows, all_scores, per_frame = [], [], [], []
    moderate_fail_drawn = False
    figure_dir = args.out_dir / "figures"
    for frame_id in args.frames:
        frame = load_frame(args.data_root, frame_id)
        cloud = args.data_root / "training" / "velodyne" / f"{frame_id}.bin"
        result, _ = inference_detector(model, str(cloud))
        boxes, scores, labels = prediction_arrays(result)
        car = labels == 0
        boxes, scores = boxes[car], scores[car]
        all_scores.extend(scores.tolist())
        gt = gt_bev_corners(frame)
        objs = car_labels(frame)
        moderate = np.array([is_moderate(o) for o in objs], dtype=bool)
        counts = points_in_gt_boxes(frame)
        for threshold in args.thresholds:
            keep = scores >= threshold
            kept_boxes, kept_scores = boxes[keep], scores[keep]
            fov = in_camera_fov(frame, kept_boxes)
            hits, pred_hits = match_centers(gt, kept_boxes, args.max_center_distance)
            ranges = np.linalg.norm(kept_boxes[:, :2], axis=1) if len(kept_boxes) else np.array([])
            row = {"frame_id": frame_id, "score_thr": threshold, "gt_cars": len(gt),
                   "gt_cars_moderate": int(moderate.sum()),
                   "pred_cars": len(kept_boxes), "pred_cars_in_fov": int(fov.sum()),
                   "pred_cars_outside_fov": int((~fov).sum()),
                   "matched_gt_cars": int(hits.sum()),
                   "matched_gt_moderate": int((hits & moderate).sum()),
                   "center_recall": hits.mean() if len(gt) else float("nan"),
                   "center_recall_moderate": hits[moderate].mean() if moderate.any() else float("nan"),
                   "unmatched_pred_in_fov": int((fov & ~pred_hits).sum()),
                   "mean_score": float(kept_scores.mean()) if len(kept_scores) else float("nan"),
                   "min_range_m": float(ranges.min()) if len(ranges) else float("nan"),
                   "max_range_m": float(ranges.max()) if len(ranges) else float("nan")}
            rows.append(row)
            if not moderate_fail_drawn and (moderate & ~hits).any():
                draw_bev(frame, kept_boxes, kept_scores,
                         figure_dir / f"fail_03_{args.tag}_moderate_car_below_thr.png",
                         args.max_center_distance,
                         f"{args.tag} {frame_id}: easy/moderate Car lost at score >= {threshold}",
                         annotate_missed=True)
                moderate_fail_drawn = True
            if threshold == args.thresholds[0]:
                draw_bev(frame, kept_boxes, kept_scores, figure_dir / f"{args.tag}_bev_{frame_id}.png",
                         args.max_center_distance, f"{args.tag} {frame_id}: score >= {threshold}")
                per_frame.append((frame_id, frame, kept_boxes, kept_scores,
                                  int((~hits).sum()), int((~fov).sum())))
                for obj, hit, n, mod in zip(objs, hits, counts, moderate):
                    center = obj.location
                    gt_rows.append({"frame_id": frame_id, "score_thr": threshold,
                                    "distance_m": float(np.hypot(center[0], center[2])),
                                    "occluded": obj.occluded, "truncated": obj.truncated,
                                    "bbox_height_px": float(obj.bbox[3] - obj.bbox[1]),
                                    "kitti_moderate": bool(mod), "lidar_points_in_box": n,
                                    "matched": bool(hit)})
    columns = list(rows[0])
    with (args.out_dir / f"{args.tag}_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / f"{args.tag}_gt_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gt_rows[0]))
        writer.writeheader()
        writer.writerows(gt_rows)

    # Failure figures, chosen from data rather than by hand.
    missed = max(per_frame, key=lambda f: f[4])
    if missed[4]:
        draw_bev(missed[1], missed[2], missed[3], figure_dir / f"fail_01_{args.tag}_occluded_cars_missed.png",
                 args.max_center_distance,
                 f"{args.tag} {missed[0]}: red GT missed (occlusion level, LiDAR points in box)",
                 annotate_missed=True)
    outside = max(per_frame, key=lambda f: f[5])
    if outside[5]:
        draw_bev(outside[1], outside[2], outside[3], figure_dir / f"fail_02_{args.tag}_unlabeled_outside_fov.png",
                 args.max_center_distance,
                 f"{args.tag} {outside[0]}: orange = confident boxes outside camera FOV (KITTI has no label)")

    thresholds = np.asarray(args.thresholds)
    def total(key: str) -> list[int]:
        return [sum(r[key] for r in rows if r["score_thr"] == t) for t in thresholds]
    total_gt = sum(r["gt_cars"] for r in rows if r["score_thr"] == thresholds[0])
    total_mod = sum(r["gt_cars_moderate"] for r in rows if r["score_thr"] == thresholds[0])
    fig, left = plt.subplots(figsize=(7, 4))
    left.plot(thresholds, total("pred_cars"), "o-", color="tab:blue", label="Predicted Car boxes (all)")
    left.plot(thresholds, total("pred_cars_in_fov"), "o:", color="tab:cyan", label="Predicted Car boxes (camera FOV)")
    left.set(xlabel="Score threshold", ylabel="Predicted Car boxes")
    right = left.twinx()
    right.plot(thresholds, np.asarray(total("matched_gt_cars")) / total_gt, "s--", color="tab:red",
               label="GT center recall (all)")
    right.plot(thresholds, np.asarray(total("matched_gt_moderate")) / max(total_mod, 1), "^--",
               color="tab:green", label="GT center recall (KITTI moderate)")
    right.set(ylabel="GT center recall", ylim=(0, 1.05))
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="lower left", fontsize=7)
    left.set_title(f"{args.tag}: score threshold trade-off, {len(args.frames)} KITTI frames")
    left.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{args.tag}_threshold_tradeoff.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_scores, bins=np.linspace(0, 1, 21), color="steelblue", edgecolor="white")
    ax.set(xlabel="Car confidence score", ylabel="Number of boxes",
           title=f"{args.tag}: scores before thresholding, {len(args.frames)} KITTI frames")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{args.tag}_score_histogram.png", dpi=150)
    plt.close(fig)

    # Latency: end-to-end (reads .bin from disk) and model-only (points already in RAM).
    frame_id = args.frames[0]
    cloud = args.data_root / "training" / "velodyne" / f"{frame_id}.bin"
    points = load_frame(args.data_root, frame_id)["points"]
    end_to_end = time_inference(model, str(cloud), args.device, args.repeats)
    in_memory = time_inference(model, points, args.device, args.repeats)
    with (args.out_dir / f"{args.tag}_latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "latency_file_ms", "latency_in_memory_ms"])
        writer.writerows((i + 1, a, b) for i, (a, b) in enumerate(zip(end_to_end, in_memory)))
    print(f"[{args.tag}] frames={len(args.frames)} rows={len(rows)} on {hardware}")
    for name, values in (("file", end_to_end), ("in-memory", in_memory)):
        print(f"  latency {name}: p50={np.percentile(values, 50):.1f} ms p95={np.percentile(values, 95):.1f} ms")


if __name__ == "__main__":
    main()
