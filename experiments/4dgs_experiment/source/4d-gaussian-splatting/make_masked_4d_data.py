from pathlib import Path

from PIL import Image

import shutil

import re

SRC_ROOT = Path("/cluster/project/cropsci/jmercoli/4dgs_project/data/4d_data")

DST_ROOT = Path("/cluster/project/cropsci/jmercoli/4dgs_project/data/4d_data_masked")

def cam_id_from_name(name):

    stem = Path(name).stem

    m = re.search(r"_([0-9]+)$", stem)

    return m.group(1) if m else None

def find_mask(mask_dir, image_name):

    stem = Path(image_name).stem

    candidates = [

        mask_dir / f"{stem}_mask_ground_truth.png",

        mask_dir / f"{stem}_mask_sam3.png",

        mask_dir / f"{stem}_mask.png",

        mask_dir / f"{stem}.png",

    ]

    for c in candidates:

        if c.exists():

            return c

    cid = cam_id_from_name(image_name)

    if cid is not None:

        matches = (

            list(mask_dir.glob(f"*_{cid}_mask_ground_truth.png")) +

            list(mask_dir.glob(f"*_{cid}_mask_sam3.png")) +

            list(mask_dir.glob(f"*_{cid}_mask.png")) +

            list(mask_dir.glob(f"*_{cid}.png"))

        )

        if matches:

            return matches[0]

    return None

if DST_ROOT.exists():

    print("Removing old:", DST_ROOT)

    shutil.rmtree(DST_ROOT)

DST_ROOT.mkdir(parents=True, exist_ok=True)

timestep_dirs = sorted([d for d in SRC_ROOT.glob("timestep_*") if d.is_dir()])

if not timestep_dirs:

    raise RuntimeError(f"No timestep_* folders found in {SRC_ROOT}")

for tdir in timestep_dirs:

    out_tdir = DST_ROOT / tdir.name

    out_img_dir = out_tdir / "images"

    out_img_dir.mkdir(parents=True, exist_ok=True)

    if not (tdir / "sparse").exists():

        raise RuntimeError(f"No sparse folder found in {tdir}")

    shutil.copytree(tdir / "sparse", out_tdir / "sparse")

    if not (tdir / "masks").exists():

        raise RuntimeError(f"No masks folder found in {tdir}")

    shutil.copytree(tdir / "masks", out_tdir / "masks")

    mask_dir = tdir / "masks"

    count = 0

    missing = []

    for img_path in sorted((tdir / "images").glob("*")):

        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:

            continue

        mask_path = find_mask(mask_dir, img_path.name)

        if mask_path is None:

            missing.append(img_path.name)

            continue

        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)

        black = Image.new("RGB", img.size, (0, 0, 0))

        masked = Image.composite(img, black, mask)

        if img_path.suffix.lower() in [".jpg", ".jpeg"]:

            masked.save(out_img_dir / img_path.name, quality=95)

        else:

            masked.save(out_img_dir / img_path.name)

        count += 1

    print(tdir.name, "masked images:", count, "missing masks:", len(missing))

    if missing:

        print("  first missing examples:", missing[:8])

print("Done:", DST_ROOT)

