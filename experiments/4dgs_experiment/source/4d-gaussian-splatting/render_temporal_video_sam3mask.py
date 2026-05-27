from argparse import ArgumentParser
from pathlib import Path
import subprocess

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render


CONFIG_PATH = "configs/plant/close10_sam3mask_3000.yaml"
CHECKPOINT_PATH = "/cluster/project/cropsci/jmercoli/4dgs_project/outputs/close10_dynamic_sam3mask_3000/chkpnt_best.pth"
OUT_DIR = Path("/cluster/project/cropsci/jmercoli/4dgs_project/outputs/close10_dynamic_sam3mask_3000/temporal_video_sam3mask_cam0")
N_FRAMES = 60


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])

    cfg = OmegaConf.load(CONFIG_PATH)

    for k, v in cfg.ModelParams.items():
        setattr(args, k, v)
    for k, v in cfg.PipelineParams.items():
        setattr(args, k, v)
    for k, v in cfg.OptimizationParams.items():
        setattr(args, k, v)

    pipe = pp.extract(args)
    opt = op.extract(args)

    gaussians = GaussianModel(
        args.sh_degree,
        gaussian_dim=int(cfg.gaussian_dim),
        time_duration=list(cfg.time_duration),
        rot_4d=bool(cfg.rot_4d),
        force_sh_3d=bool(cfg.force_sh_3d),
        sh_degree_t=2 if args.eval_shfs_4d else 0,
    )

    scene = Scene(
        args,
        gaussians,
        load_iteration=None,
        shuffle=False,
        num_pts=int(cfg.num_pts),
        num_pts_ratio=float(cfg.num_pts_ratio),
        time_duration=list(cfg.time_duration),
    )

    print("Loading checkpoint:", CHECKPOINT_PATH)
    model_params, iteration = torch.load(CHECKPOINT_PATH)
    gaussians.restore(model_params, opt)
    print("Loaded checkpoint iteration:", iteration)

    # Use raw Camera objects, not CameraDataset tuples.
    # Pick one test camera pose and keep it fixed while time changes.
    if len(scene.test_cameras[1.0]) > 0:
        cam = scene.test_cameras[1.0][0]
        print("Using test camera:", cam.image_name)
    else:
        cam = scene.train_cameras[1.0][0]
        print("Using train camera:", cam.image_name)

    cam = cam.cuda()

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    t0, t1 = float(cfg.time_duration[0]), float(cfg.time_duration[1])
    times = torch.linspace(t0, t1, N_FRAMES).tolist()

    with torch.no_grad():
        for i, t in enumerate(times):
            cam.timestamp = float(t)
            print(f"Rendering frame {i:04d}, t={t:.4f}")
            pkg = render(cam, gaussians, pipe, background)
            image = pkg["render"].clamp(0, 1)
            save_image(image, frames_dir / f"frame_{i:04d}.png")
            torch.cuda.synchronize()

    mp4_path = OUT_DIR / "temporal_cam0.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "15",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-pix_fmt", "yuv420p",
        str(mp4_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Saved video:", mp4_path)


if __name__ == "__main__":
    main()
