from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = EXPERIMENT_ROOT / "scripts" / "train_mask_loss_3dgs.py"


def patch_dataset_readers(repo: Path, test_count: int) -> bool:
    path = repo / "scene" / "dataset_readers.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "alpha_mask_path: str | None" not in text and "alpha_mask_path: str" not in text:
        text = text.replace(
            "    mask_paths: List[str]\n",
            "    mask_paths: List[str]\n    alpha_mask_path: str | None\n",
        )

    if "masks_binary_active" not in text:
        text = text.replace(
            '        image_path = os.path.join(images_folder, os.path.basename(extr.name))\n'
            '        image_name = os.path.basename(image_path).split(".")[0]\n'
            "        image = Image.open(image_path)\n",
            '        image_path = os.path.join(images_folder, os.path.basename(extr.name))\n'
            '        image_name = os.path.basename(image_path).split(".")[0]\n'
            "        image = Image.open(image_path)\n\n"
            "        alpha_mask_path = os.path.join(\n"
            "            os.path.dirname(images_folder),\n"
            '            "masks_binary_active",\n'
            '            image_name + "_mask_ground_truth.png"\n'
            "        )\n"
            "        alpha_mask_path = os.path.normpath(alpha_mask_path)\n"
            "        if not os.path.exists(alpha_mask_path):\n"
            "            alpha_mask_path = None\n",
        )

    if "alpha_mask_path=alpha_mask_path" not in text:
        text = text.replace(
            "        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,\n"
            "                              image_path=image_path, image_name=image_name, width=width, height=height,\n"
            "                              bbox_path=bbox_path, mask_paths=mask_paths)\n",
            "        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,\n"
            "                              image_path=image_path, image_name=image_name, width=width, height=height,\n"
            "                              bbox_path=bbox_path, mask_paths=mask_paths, alpha_mask_path=alpha_mask_path)\n",
        )

    split_marker = "train_cam_infos = cam_infos[:-test_count]"
    if split_marker not in text:
        pattern = re.compile(
            r"    if eval:\n(?:.*?\n)*?    else:\n        train_cam_infos = cam_infos\n        test_cam_infos = \[\]\n",
            re.MULTILINE,
        )
        replacement = (
            "    if eval:\n"
            f"        test_count = {test_count}\n"
            "        train_cam_infos = cam_infos[:-test_count]\n"
            "        test_cam_infos = cam_infos[-test_count:]\n"
            "        print(f\"Train Cam list with {len(train_cam_infos)} cams: {[cam.image_name for cam in train_cam_infos]}\")\n"
            "        print(f\"Test Cam list with {len(test_cam_infos)} cams: {[cam.image_name for cam in test_cam_infos]}\")\n"
            "    else:\n"
            "        train_cam_infos = cam_infos\n"
            "        test_cam_infos = []\n"
        )
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not patch eval split in {path}")

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_camera_utils(repo: Path) -> bool:
    path = repo / "utils" / "camera_utils.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "from PIL import Image" not in text:
        text = text.replace("import numpy as np\n", "import numpy as np\nfrom PIL import Image\n")
    if "import torch" not in text:
        text = text.replace("from PIL import Image\n", "from PIL import Image\nimport torch\n")

    if 'getattr(cam_info, "alpha_mask_path", None)' not in text:
        pattern = re.compile(
            r"    gt_image\s*=\s*resized_image_rgb\[:3,\s*\.\.\.\]\n"
            r"    loaded_mask\s*=\s*None\n\n"
            r"    if resized_image_rgb\.shape\[(?:0|1)\]\s*==\s*4:\n"
            r"        loaded_mask\s*=\s*resized_image_rgb\[3:4,\s*\.\.\.\]\n",
        )
        replacement = (
            "    gt_image = resized_image_rgb[:3, ...]\n"
            "    loaded_mask = None\n\n"
            "    if resized_image_rgb.shape[0] == 4:\n"
            "        loaded_mask = resized_image_rgb[3:4, ...]\n\n"
            "    if getattr(cam_info, \"alpha_mask_path\", None) is not None:\n"
            "        alpha_pil = Image.open(cam_info.alpha_mask_path).convert(\"L\")\n"
            "        alpha_resized = alpha_pil.resize(resolution, Image.NEAREST)\n"
            "        alpha_np = np.array(alpha_resized)\n"
            "        alpha_np = (alpha_np > 128).astype(np.float32)\n"
            "        loaded_mask = torch.from_numpy(alpha_np).unsqueeze(0)\n"
        )
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not patch mask loading in {path}")

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_cameras(repo: Path) -> bool:
    path = repo / "scene" / "cameras.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "self.gt_alpha_mask" not in text:
        old = (
            "        if gt_alpha_mask is not None:\n"
            "            self.original_image *= gt_alpha_mask.to(self.data_device)\n"
            "        else:\n"
            "            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)\n"
        )
        new = (
            "        if gt_alpha_mask is not None:\n"
            "            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)\n"
            "        else:\n"
            "            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)\n\n"
            "        self.original_image *= self.gt_alpha_mask\n"
        )
        if old not in text:
            raise RuntimeError(f"Could not find alpha-mask block in {path}")
        text = text.replace(old, new)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Wheat-3DGS for masked GT and loss experiments.")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--test-count", type=int, default=5)
    parser.add_argument("--no-copy-train-script", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / "scene").exists():
        raise FileNotFoundError(f"Does not look like a Wheat-3DGS checkout: {repo}")

    changed = {
        "scene/dataset_readers.py": patch_dataset_readers(repo, args.test_count),
        "utils/camera_utils.py": patch_camera_utils(repo),
        "scene/cameras.py": patch_cameras(repo),
    }

    if not args.no_copy_train_script:
        shutil.copy2(TRAIN_SOURCE, repo / "train_mask_loss_3dgs.py")
        changed["train_mask_loss_3dgs.py"] = True

    for relpath, did_change in changed.items():
        print(f"{relpath}: {'updated' if did_change else 'already ok'}")


if __name__ == "__main__":
    main()
