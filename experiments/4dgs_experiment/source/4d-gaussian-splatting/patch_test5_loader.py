from pathlib import Path

repo = Path("/cluster/project/cropsci/jmercoli/4dgs_project/repos/4d-gaussian-splatting")

# ============================================================
# Patch scene/dataset_readers.py
# Add readTest5ColmapSceneInfo()
# ============================================================

p = repo / "scene/dataset_readers.py"
text = p.read_text()

if "def readTest5ColmapSceneInfo" not in text:
    insert = r'''

def readTest5ColmapSceneInfo(path, images, eval, llffhold=8, num_pts_ratio=1.0):
    """
    Custom loader for Juliane's plant 4D test set.

    Expected structure:
        path/
          timestep_0000/
            images/
            sparse/0/{cameras.bin, images.bin, points3D.bin}
          timestep_0378/
          timestep_0762/
          timestep_1140/
          timestep_1518/

    We read each timestep as a COLMAP scene, assign normalized timestamp in [0,1],
    combine all cameras, and initialize the point cloud from the first timestep.
    """
    timestep_dirs = sorted([
        d for d in Path(path).iterdir()
        if d.is_dir() and d.name.startswith("timestep_")
    ])

    if len(timestep_dirs) == 0:
        raise RuntimeError(f"No timestep_* folders found under {path}")

    print("Found Test5 dynamic COLMAP dataset:")
    for d in timestep_dirs:
        print("  ", d.name)

    all_train = []
    all_test = []

    # normalize timestamps from 0 to 1
    if len(timestep_dirs) == 1:
        times = [0.0]
    else:
        times = [i / (len(timestep_dirs) - 1) for i in range(len(timestep_dirs))]

    reference_scene_info = None

    for t_idx, (tdir, timestamp) in enumerate(zip(timestep_dirs, times)):
        tdir = str(tdir)

        try:
            cameras_extrinsic_file = os.path.join(tdir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(tdir, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except Exception:
            cameras_extrinsic_file = os.path.join(tdir, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(tdir, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = "images" if images is None else images
        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(tdir, reading_dir)
        )
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

        # Assign timestamp and unique uid/image_name to avoid collisions
        stamped = []
        for local_idx, c in enumerate(cam_infos):
            stamped.append(CameraInfo(
                uid=t_idx * 10000 + local_idx,
                R=c.R,
                T=c.T,
                FovY=c.FovY,
                FovX=c.FovX,
                image=c.image,
                depth=c.depth,
                image_path=c.image_path,
                image_name=f"{Path(tdir).name}_{c.image_name}",
                width=c.width,
                height=c.height,
                timestamp=float(timestamp),
                fl_x=c.fl_x,
                fl_y=c.fl_y,
                cx=c.cx,
                cy=c.cy,
            ))

        if eval:
            train_t = [c for idx, c in enumerate(stamped) if idx % llffhold != 0]
            test_t = [c for idx, c in enumerate(stamped) if idx % llffhold == 0]
        else:
            train_t = stamped
            test_t = []

        all_train.extend(train_t)
        all_test.extend(test_t)

        if reference_scene_info is None:
            # Build/read point cloud from first timestep only
            ply_path = os.path.join(tdir, "sparse/0/points3D.ply")
            bin_path = os.path.join(tdir, "sparse/0/points3D.bin")
            txt_path = os.path.join(tdir, "sparse/0/points3D.txt")
            if not os.path.exists(ply_path):
                print("Converting first timestep points3D to .ply for initialization.")
                try:
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                except Exception:
                    xyz, rgb, _ = read_points3D_text(txt_path)
                storePly(ply_path, xyz, rgb)
            try:
                pcd = fetchPly(ply_path)
            except Exception:
                pcd = None
            reference_scene_info = (pcd, ply_path)

    nerf_normalization = getNerfppNorm(all_train)

    pcd, ply_path = reference_scene_info

    if num_pts_ratio > 1.001 and pcd is not None:
        num_pts = int((num_pts_ratio - 1) * pcd.points.shape[0])
        mean_xyz = pcd.points.mean(axis=0)
        min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
        max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
        xyz = np.concatenate([
            pcd.points,
            np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz
        ], axis=0)
        colors = np.concatenate([
            pcd.colors,
            SH2RGB(np.random.random((num_pts, 3)) / 255.0)
        ], axis=0)
        normals = np.concatenate([
            pcd.normals,
            np.zeros((num_pts, 3))
        ], axis=0)
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

    print(f"Total train cameras: {len(all_train)}")
    print(f"Total test cameras: {len(all_test)}")
    print(f"Timestamps: {times}")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=all_train,
        test_cameras=all_test,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    return scene_info
'''
    # Insert before sceneLoadTypeCallbacks
    marker = "sceneLoadTypeCallbacks = {"
    if marker not in text:
        raise RuntimeError("Could not find sceneLoadTypeCallbacks marker")
    text = text.replace(marker, insert + "\n" + marker)

# Add callback
if '"Test5Colmap"' not in text:
    text = text.replace(
        'sceneLoadTypeCallbacks = {\n    "Colmap": readColmapSceneInfo,\n    "Blender" : readNerfSyntheticInfo\n}',
        'sceneLoadTypeCallbacks = {\n    "Colmap": readColmapSceneInfo,\n    "Blender" : readNerfSyntheticInfo,\n    "Test5Colmap": readTest5ColmapSceneInfo\n}'
    )

p.write_text(text)
print("✅ Patched scene/dataset_readers.py with Test5Colmap loader")


# ============================================================
# Patch scene/__init__.py
# Recognize source_path containing timestep_* folders
# ============================================================

p = repo / "scene/__init__.py"
text = p.read_text()

if "Found timestep_* folders, assuming Test5 dynamic COLMAP dataset!" not in text:
    old = '''        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        else:
            assert False, "Could not recognize scene type!"
'''

    new = '''        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif any(name.startswith("timestep_") and os.path.isdir(os.path.join(args.source_path, name)) for name in os.listdir(args.source_path)):
            print("Found timestep_* folders, assuming Test5 dynamic COLMAP dataset!")
            scene_info = sceneLoadTypeCallbacks["Test5Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        else:
            assert False, "Could not recognize scene type!"
'''

    if old not in text:
        raise RuntimeError("Could not find scene type detection block in scene/__init__.py")
    text = text.replace(old, new)

p.write_text(text)
print("✅ Patched scene/__init__.py to recognize timestep_* datasets")
