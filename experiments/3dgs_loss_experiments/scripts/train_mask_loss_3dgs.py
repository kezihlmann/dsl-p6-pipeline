from __future__ import annotations

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from gaussian_renderer import render
from scene import GaussianModel, Scene
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


LOSS_MODES = {
    "baseline",
    "masked_rgb",
    "alpha",
    "rgb_alpha",
    "foreground_background",
}


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if value.ndim == 3 and mask.ndim == 3 and mask.shape[0] == 1 and value.shape[0] != 1:
        mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def masked_l1(image: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(torch.abs(image - gt), mask)


def alpha_loss(render_pkg: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    pred_alpha = render_pkg.get("alpha")
    if pred_alpha is None:
        pred_alpha = render_pkg["render"].amax(dim=0, keepdim=True)
    if pred_alpha.ndim == 2:
        pred_alpha = pred_alpha.unsqueeze(0)
    pred_alpha = pred_alpha.clamp(1e-5, 1.0 - 1e-5)
    return F.binary_cross_entropy(pred_alpha, mask.float())


def compute_loss(image: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, render_pkg: dict[str, torch.Tensor], opt) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rgb_l1 = l1_loss(image, gt)
    rgb_ssim = 1.0 - ssim(image, gt)
    base_rgb = (1.0 - opt.lambda_dssim) * rgb_l1 + opt.lambda_dssim * rgb_ssim

    components = {
        "rgb_l1": rgb_l1.detach(),
        "rgb_ssim": rgb_ssim.detach(),
        "alpha_bce": torch.zeros((), device=image.device),
        "background_alpha": torch.zeros((), device=image.device),
    }

    if opt.loss_mode == "baseline":
        return base_rgb, components

    if opt.loss_mode == "masked_rgb":
        loss = masked_l1(image, gt, mask)
        return loss, components

    a_loss = alpha_loss(render_pkg, mask)
    components["alpha_bce"] = a_loss.detach()

    if opt.loss_mode == "alpha":
        return base_rgb + opt.silhouette_weight * a_loss, components

    if opt.loss_mode == "rgb_alpha":
        fg_rgb = masked_l1(image, gt, mask)
        return fg_rgb + opt.silhouette_weight * a_loss, components

    if opt.loss_mode == "foreground_background":
        fg_rgb = masked_l1(image, gt, mask)
        pred_alpha = render_pkg.get("alpha")
        if pred_alpha is None:
            pred_alpha = image.amax(dim=0, keepdim=True)
        bg_mask = 1.0 - mask.float()
        bg_alpha = masked_mean(pred_alpha, bg_mask)
        components["background_alpha"] = bg_alpha.detach()
        return fg_rgb + opt.silhouette_weight * a_loss + opt.background_loss_weight * bg_alpha, components

    raise ValueError(f"Unknown loss_mode: {opt.loss_mode}")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID", str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def training_report(tb_writer, iteration, loss, components, elapsed, testing_iterations, scene: Scene, render_func, render_args):
    if tb_writer:
        tb_writer.add_scalar("train_loss/total", loss.item(), iteration)
        for name, value in components.items():
            tb_writer.add_scalar(f"train_loss/{name}", value.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
    )

    for config in validation_configs:
        if not config["cameras"]:
            continue
        l1_test = 0.0
        psnr_test = 0.0
        for idx, viewpoint in enumerate(config["cameras"]):
            image = torch.clamp(render_func(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if tb_writer and idx < 5:
                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
        psnr_test /= len(config["cameras"])
        l1_test /= len(config["cameras"])
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    print("Length of viewpoint stack", len(viewpoint_stack))
    print("Loss mode:", opt.loss_mode)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.gt_alpha_mask.cuda().float()
        loss, components = compute_loss(image, gt_image, mask, render_pkg, opt)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 500 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Num_points": len(gaussians.get_xyz)})
                progress_bar.update(500)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, loss, components, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    parser = ArgumentParser(description="Masked 3DGS loss experiment training script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 15000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 15000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--loss_mode", choices=sorted(LOSS_MODES), default="baseline")
    parser.add_argument("--silhouette_weight", type=float, default=0.1)
    parser.add_argument("--background_loss_weight", type=float, default=0.01)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    opt = op.extract(args)
    opt.loss_mode = args.loss_mode
    opt.silhouette_weight = args.silhouette_weight
    opt.background_loss_weight = args.background_loss_weight
    training(lp.extract(args), opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")
