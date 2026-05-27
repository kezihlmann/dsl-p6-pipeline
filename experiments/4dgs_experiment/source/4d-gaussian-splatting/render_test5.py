from argparse import ArgumentParser
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render


CONFIG_PATH = "configs/plant/test5_3000.yaml"
CHECKPOINT_PATH = "/cluster/project/cropsci/jmercoli/4dgs_project/outputs/test5_dynamic_3000/chkpnt_best.pth"
OUT_DIR = Path("/cluster/project/cropsci/jmercoli/4dgs_project/outputs/test5_dynamic_3000/render_debug")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = ArgumentParser(description="Render 4DGS test5 debug views")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args([])
    cfg = OmegaConf.load(CONFIG_PATH)

    # Fill args from YAML config
    for k, v in cfg.ModelParams.items():
        setattr(args, k, v)
    for k, v in cfg.PipelineParams.items():
        setattr(args, k, v)
    for k, v in cfg.OptimizationParams.items():
        setattr(args, k, v)

    gaussian_dim = int(cfg.gaussian_dim)
    time_duration = list(cfg.time_duration)
    rot_4d = bool(cfg.rot_4d)
    force_sh_3d = bool(cfg.force_sh_3d)

    pipe = pp.extract(args)
    opt = op.extract(args)

    gaussians = GaussianModel(
        args.sh_degree,
        gaussian_dim=gaussian_dim,
        time_duration=time_duration,
        rot_4d=rot_4d,
        force_sh_3d=force_sh_3d,
        sh_degree_t=2 if args.eval_shfs_4d else 0,
    )

    # Build scene/cameras first
    scene = Scene(
        args,
        gaussians,
        load_iteration=None,
        shuffle=False,
        num_pts=int(cfg.num_pts),
        num_pts_ratio=float(cfg.num_pts_ratio),
        time_duration=time_duration,
    )

    # Then restore trained checkpoint, otherwise Scene initialization overwrites gaussians
    print("Loading checkpoint:", CHECKPOINT_PATH)
    model_params, iteration = torch.load(CHECKPOINT_PATH)
    gaussians.restore(model_params, opt)
    print("Loaded checkpoint iteration:", iteration)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    print("Train cameras:", len(train_cams))
    print("Test cameras:", len(test_cams))

    selected = []
    for i in range(min(10, len(test_cams))):
        selected.append(("test", i, test_cams[i]))
    for i in range(min(5, len(train_cams))):
        selected.append(("train", i, train_cams[i]))

    with torch.no_grad():
        for split, idx, item in selected:
            # CameraDataset may return either a Camera directly or a tuple containing it.
            # Find the tuple element that is actually the Camera object.
            if isinstance(item, tuple):
                cam = None
                for x in item:
                    if hasattr(x, "FoVx") and hasattr(x, "FoVy"):
                        cam = x
                        break
                if cam is None:
                    raise RuntimeError(f"Could not find Camera object in tuple: {[type(x) for x in item]}")
            else:
                cam = item

            result = render(cam, gaussians, pipe, background)
            image = result["render"].clamp(0, 1)

            timestamp = float(cam.timestamp)
            safe_name = cam.image_name.replace("/", "_")

            pred_path = OUT_DIR / f"{split}_{idx:03d}_t{timestamp:.2f}_{safe_name}_pred.png"
            gt_path = OUT_DIR / f"{split}_{idx:03d}_t{timestamp:.2f}_{safe_name}_gt.png"

            save_image(image, pred_path)
            save_image(cam.original_image.cuda().clamp(0, 1), gt_path)

            print("Saved:", pred_path)
            print("Saved:", gt_path)


if __name__ == "__main__":
    main()
