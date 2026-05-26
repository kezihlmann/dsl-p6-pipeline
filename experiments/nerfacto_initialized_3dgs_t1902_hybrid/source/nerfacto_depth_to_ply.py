from pathlib import Path
import argparse
import numpy as np
import torch
import open3d as o3d

from nerfstudio.utils.eval_utils import eval_setup


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--accum-thresh", type=float, default=0.005)
    parser.add_argument("--rays-per-chunk", type=int, default=512)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading config: {args.config}", flush=True)
    config, pipeline, checkpoint_path, step = eval_setup(args.config, test_mode="test")
    device = pipeline.device
    pipeline.model.eval()

    dataset = pipeline.datamanager.eval_dataset if args.split == "test" else pipeline.datamanager.train_dataset
    cameras = dataset.cameras.to(device)

    all_points = []
    all_colors = []

    print(f"Using split={args.split}, cameras={len(cameras)}", flush=True)

    with torch.no_grad():
        for cam_idx in range(len(cameras)):
            print(f"Rendering camera {cam_idx+1}/{len(cameras)} in chunks", flush=True)

            cam = cameras[cam_idx : cam_idx + 1]
            ray_bundle = cam.generate_rays(camera_indices=0)
            total_rays = int(ray_bundle.origins.numel() // 3)

            cam_points = []
            cam_colors = []

            for start in range(0, total_rays, args.rays_per_chunk):
                end = min(start + args.rays_per_chunk, total_rays)

                rb = ray_bundle.get_row_major_sliced_ray_bundle(start, end)
                outputs = pipeline.model.forward(rb)

                rgb = outputs["rgb"]
                depth = outputs["depth"]
                accum = outputs["accumulation"]

                if depth.ndim > 1:
                    depth = depth[..., 0]
                if accum.ndim > 1:
                    accum = accum[..., 0]

                valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(accum) & (accum > args.accum_thresh)

                if valid.any():
                    origins = rb.origins
                    dirs = rb.directions

                    pts = origins[valid] + dirs[valid] * depth[valid, None]
                    cols = rgb[valid]

                    cam_points.append(to_np(pts))
                    cam_colors.append(to_np(cols))

                del rb, outputs, rgb, depth, accum
                torch.cuda.empty_cache()

            if cam_points:
                pts = np.concatenate(cam_points, axis=0)
                cols = np.concatenate(cam_colors, axis=0)

                # If too many points from this camera, randomly keep a subset.
                per_cam_cap = max(1000, args.max_points // max(1, len(cameras)))
                if len(pts) > per_cam_cap:
                    rng = np.random.default_rng(cam_idx)
                    keep = rng.choice(len(pts), size=per_cam_cap, replace=False)
                    pts = pts[keep]
                    cols = cols[keep]

                print(f"Camera {cam_idx+1}: kept {len(pts)} points", flush=True)
                all_points.append(pts)
                all_colors.append(cols)
            else:
                print(f"Camera {cam_idx+1}: no valid points at threshold {args.accum_thresh}", flush=True)

    if not all_points:
        raise RuntimeError(
            f"No points generated. Try lowering --accum-thresh, e.g. 0.001 or 0.0005."
        )

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if len(points) > args.max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    colors = np.clip(colors, 0.0, 1.0)

    print(f"Saving {len(points)} points to {args.out}", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    ok = o3d.io.write_point_cloud(str(args.out), pcd)

    print(f"Done. write_success={ok}", flush=True)


if __name__ == "__main__":
    main()
