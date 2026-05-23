# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pdb
import cv2
import trimesh
import pickle
from collections import deque, defaultdict
import torch
import pycocotools.mask as mask_utils
import glob
import random
# fix seed
random.seed(42)

def rle_to_mask(rle):
    rle = dict(rle)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle).astype(bool)

def res_to_instance_map(res):
    rles = res["masks_rle"]
    obj_ids = res["obj_ids"]

    if torch.is_tensor(obj_ids):
        obj_ids = obj_ids.cpu().numpy()

    if len(rles) == 0:
        return np.zeros((192, 256), dtype=np.int32)

    first_mask = rle_to_mask(rles[0])
    H, W = first_mask.shape

    instance_map = np.zeros((H, W), dtype=np.int32)

    for rle, obj_id in zip(rles, obj_ids):
        mask = rle_to_mask(rle)
        instance_map[mask] = int(obj_id)

    return instance_map

def to_bool_mask(mask):
    mask = torch.as_tensor(mask)

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.shape}")

    return mask.bool()

def merged_groups_to_panoptic_full(
    merged_groups,
    num_frames,
    image_size=None,
    background_id=0,
    panoptic_id_start=1,
    output="torch",
):
    """
    Convert merged_groups to a dense list of panoptic maps for all frames.

    Args:
        merged_groups: dict[group_id] = {
            "members": [...],
            "targets": [frame ids...],
            "masks":   [HxW bool masks...]
        }
        num_frames: total number of frames in the video
        image_size: (H, W). Optional if can be inferred from first non-empty mask.
        background_id: background label
        panoptic_id_start: first panoptic instance id
        output: "torch" or "numpy"
    """
    sorted_group_ids = sorted(merged_groups.keys())
    group_to_panoptic_id = {
        gid: panoptic_id_start + i
        for i, gid in enumerate(sorted_group_ids)
    }

    # Infer H, W if not provided
    H = W = None
    if image_size is not None:
        H, W = image_size
    else:
        for gid in sorted_group_ids:
            for mask in merged_groups[gid]["masks"]:
                mask = to_bool_mask(mask)
                if mask.numel() > 0:
                    H, W = mask.shape
                    break
            if H is not None:
                break

    if H is None or W is None:
        raise ValueError("Could not infer image size. Please provide image_size=(H, W).")

    # Initialize all frames to background
    panoptic_list = [
        torch.full((H, W), background_id, dtype=torch.long)
        for _ in range(num_frames)
    ]

    # Collect masks per frame
    frame_entries = defaultdict(list)
    for gid in sorted_group_ids:
        pano_id = group_to_panoptic_id[gid]
        targets = merged_groups[gid]["targets"]
        masks = merged_groups[gid]["masks"]

        for frame_id, mask in zip(targets, masks):
            frame_id = int(frame_id)
            if not (0 <= frame_id < num_frames):
                continue

            mask = to_bool_mask(mask)
            if not mask.any():
                continue

            frame_entries[frame_id].append((pano_id, mask))

    # Rasterize each frame
    for frame_id, entries in frame_entries.items():
        # larger first, smaller overwrite later
        entries = sorted(entries, key=lambda x: x[1].sum().item(), reverse=True)

        pano = panoptic_list[frame_id]
        for pano_id, mask in entries:
            pano[mask] = pano_id

    if output == "numpy":
        panoptic_list = [x.cpu().numpy() for x in panoptic_list]
    elif output != "torch":
        raise ValueError(f"Unsupported output={output}")

    return panoptic_list, group_to_panoptic_id

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    objects: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    num_classes: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readScannetCameras(path, panoptic_list_train, images_folder, method, data_source, eval_list=None):
    # pdb.set_trace()
    cam_infos = []
    if data_source == 'dslr':
        with open(os.path.join(path, 'nerfstudio/transforms_undistorted.json'), 'r') as f:
            cameras = json.load(f)
    elif data_source == 'iphone':
        with open(os.path.join(path, 'transforms_imu.json'), 'r') as f:
            cameras = json.load(f)
    height = cameras['h']
    width = cameras['w']
    FovY = focal2fov(cameras['fl_y'], height)
    FovX = focal2fov(cameras['fl_x'], width)
    count = 0
    if data_source == 'dslr':
        frames = cameras['frames'] + cameras['test_frames']
        frames = sorted(frames, key=lambda x: x['file_path'])
    elif data_source == 'iphone':
        frames = cameras['frames']
    for idx, d in enumerate(frames):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        if data_source == 'iphone':
            if eval_list is None:
                if idx % 10 != 0:
                    continue
            else:
                if idx not in eval_list:
                    continue
        count += 1
        sys.stdout.write("Reading camera {}".format(count))
        sys.stdout.flush()
        if data_source == 'dslr':
            c2w = np.array(d['transform_matrix'])
            c2w[:3, 1:3] *= -1
        elif data_source == 'iphone':
            c2w = np.array(d['transform_matrix'])
            c2w = c2w[[1,0,2,3]]
            c2w[2] *= -1
            c2w[:,1] *= -1
            c2w[:,2] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        
        if data_source == 'dslr':
            image_path = os.path.join(images_folder, d['file_path']).replace('images/', 'resized_undistorted_images/')
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path) if os.path.exists(image_path) else None
            image = image.resize((512, 336))
        elif data_source == 'iphone':
            image_path = os.path.join(images_folder, d['file_path']).replace('images/', 'rgb/')
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path) if os.path.exists(image_path) else None
            image = image.resize((512, 384))
        height, width = image.size[1], image.size[0]
        if method in 'ours':
            if data_source == 'dslr':
                objects = panoptic_list_train[idx]
                objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            elif data_source == 'iphone':
                if eval_list is None:
                    objects = panoptic_list_train[int(image_name.split('_')[1])//10]
                else:
                    objects = Image.open(image_path.replace('test_Scannetppv2', 'scannetpp_panoptica_full').replace('iphone', 'panoptic').replace('/rgb/', '/').replace('.jpg', '.png'))
                    objects = np.asarray(objects)[:,:,1]
                    objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        elif method == 'sam3':
            if data_source == 'dslr':
                objects = panoptic_list_train[idx]
                objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            elif data_source == 'iphone':
                if eval_list is None:
                    objects = panoptic_list_train[int(image_name.split('_')[1])//10]
                else:
                    objects = Image.open(image_path.replace('test_Scannetppv2', 'scannetpp_panoptica_full').replace('iphone', 'panoptic').replace('/rgb/', '/').replace('.jpg', '.png'))
                    objects = np.asarray(objects)[:,:,1]
                    objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        elif method == 'gt':
            if data_source == 'dslr':
                objects = Image.open(image_path.replace('Scannetpp/NVS/data', 'scannetpp_panoptica_full_dslr').replace('/dslr/', '/panoptic/').replace('/resized_undistorted_images/', '/').replace('.JPG', '.png'))
                objects = np.asarray(objects)[:,:,1]
                objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            elif data_source == 'iphone':
                objects = Image.open(image_path.replace('test_Scannetppv2', 'scannetpp_panoptica_full').replace('iphone', 'panoptic').replace('/rgb/', '/').replace('.jpg', '.png'))
                objects = np.asarray(objects)[:,:,1]
                objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        elif method == 'panst3r':
            if data_source == 'dslr':
                if idx in panoptic_list_train.keys():
                    objects = panoptic_list_train[idx]
                else:
                    objects = None
            elif data_source == 'iphone':
                if eval_list is None:
                    if int(image_name.split('_')[1])//10 in panoptic_list_train.keys():
                        objects = panoptic_list_train[int(image_name.split('_')[1])//10]
                    else:
                        objects = None
                else:
                    objects = Image.open(image_path.replace('test_Scannetppv2', 'scannetpp_panoptica_full').replace('iphone', 'panoptic').replace('/rgb/', '/').replace('.jpg', '.png'))
                    objects = np.asarray(objects)[:,:,1]
                    objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        uid = d['file_path'].split('_')[-1].split('.')[0]
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readScannetCamerasFreeViewpoint(path, scene_name, images_folder, eval_list=None):
    cam_infos = []
    with open(f'/fs/nexus-projects/Audio23/backbones/worldexplorer/scenes/river_10.0_20251004_170903/img2trajvid/camera_paths/{scene_name}_dslr.json', 'r') as f:
        cams = json.load(f)['camera_path']
    # pdb.set_trace()
    for idx, d in enumerate(range(len(cams))):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}".format(idx))
        sys.stdout.flush()
        
        height = 336
        width = 512

        uid = idx
        c2w = np.array(cams[idx]['camera_to_world']).reshape(4,4)
        w2c = np.linalg.inv(c2w)
        w2c[1] *= -1
        w2c[2] *= -1
        # pdb.set_trace()
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3,3]
        FovY = np.deg2rad(60.0)
        FovX = 2 * np.arctan(np.tan(FovY / 2) * (width / height))
        image_name = str(idx).zfill(3)
        image_path = '/fs/nexus-projects/3D_r2s/gaussian-grouping/empty.png'

        image = Image.open(image_path)
        image = image.resize((512, 336))
        height, width = image.size[1], image.size[0]
        objects = Image.open(image_path)
        objects = np.asarray(objects)[:,:,1]
        objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name)).replace('images/', 'rgb/')
        image_name = os.path.basename(image_path).split(".")[0]
        # pdb.set_trace()
        image = Image.open(image_path) if os.path.exists(image_path) else None
        image = image.resize((512, 336))
        height, width = image.size[1], image.size[0]

        # pdb.set_trace()
        # object_path = os.path.join(objects_folder, image_name + '.png')
        # objects = Image.open(object_path) if os.path.exists(object_path) else None

        objects = Image.open(image_path.replace('test_Scannetppv2', 'scannetpp_panoptica_full').replace('iphone', 'panoptic').replace('/rgb/', '/').replace('.jpg', '.png'))
        objects = np.asarray(objects)[:,:,1]
        objects = cv2.resize(objects, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        # pdb.set_trace()

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, object_path, llffhold=8, n_views=100, random_init=False, train_split=False, free_viewpoint=False, method='ours', data_source='dslr'):
    # pdb.set_trace()
    scene_name = path.split('/')[-2]

    panoptic_list_train = []
    if method == 'ours':
        if data_source == 'dslr':
            with open(f'/fs/nexus-projects/3D_r2s/dataset/result_dslr/{scene_name}/run_shortest_graph/step01.pkl', 'rb') as f:
                final_predictions = pickle.load(f)
            with open(f'/fs/nexus-projects/3D_r2s/dataset/result_dslr/{scene_name}/run_shortest_graph/step02.pkl', 'rb') as f:
                final_predictions_pan = pickle.load(f)
        elif data_source == 'iphone':
            with open(f'/fs/nexus-projects/3D_r2s/phuc_shared/result_scannetpp_final/{scene_name}/run_shortest_graph/step01.pkl', 'rb') as f:
                final_predictions = pickle.load(f)
            with open(f'/fs/nexus-projects/3D_r2s/phuc_shared/result_scannetpp_final/{scene_name}/run_shortest_graph/step02.pkl', 'rb') as f:
                final_predictions_pan = pickle.load(f)
        panoptic_list_train, _ = merged_groups_to_panoptic_full(
            final_predictions_pan,
            num_frames=len(final_predictions["images"]),
            background_id=0,
            panoptic_id_start=1,
            output="numpy",
            )
        num_instances = max(int(x.max()) for x in panoptic_list_train)+1
    elif method == 'panst3r':
        if data_source == 'dslr':
            with open(f'/fs/nexus-projects/3D_r2s/dataset/result_dslr/{scene_name}/run_shortest_graph/step01.pkl', 'rb') as f:
                final_predictions = pickle.load(f)
        elif data_source == 'iphone':
            with open(f'/fs/nexus-projects/3D_r2s/phuc_shared/result_scannetpp_final/{scene_name}/run_shortest_graph/step01.pkl', 'rb') as f:
                final_predictions = pickle.load(f)
        instance_ids_raw = final_predictions['pan']
        instance_ids_final = instance_ids_raw.copy()
        count = 0
        panoptic_list_train = {}
        interval = (len(final_predictions['cluster_index'])-1)//2
        for cluster_idx, cluster_views in enumerate(final_predictions['cluster_index']):
            if cluster_idx % interval != 0:
                continue
            instance_ids_raw_cluster = instance_ids_raw[cluster_views]
            # instance_ids_final[cluster_views][instance_ids_final[cluster_views] != 0] += count
            tmp = instance_ids_final[cluster_views]
            tmp[tmp != 0] += count
            instance_ids_final[cluster_views] = tmp
            for view_id in cluster_views:
                panoptic_list_train[view_id] = instance_ids_final[view_id].astype(np.int64)
            count += instance_ids_raw_cluster.max()
        num_instances = max(int(x.max()) for key, x in panoptic_list_train.items())+1
    elif method == 'sam3':
        if data_source == 'dslr':
            sam3_data = torch.load(f"/fs/nexus-projects/3D_r2s/dataset/Scannetpp/sam3_scannetpp_mask_track_dslr/{scene_name}.pth")
        elif data_source == 'iphone':
            sam3_data = torch.load(f"/fs/nexus-projects/3D_r2s/dataset/Scannetpp/sam3_scannetpp_mask_track/{scene_name}.pth")
        panoptic_list_train = []
        for fname in sorted(sam3_data["results"].keys()):
            res = sam3_data["results"][fname]
            inst_map = res_to_instance_map(res)
            H = 336 if data_source == 'dslr' else 384
            inst_map = cv2.resize(inst_map, (512, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            panoptic_list_train.append(inst_map)
        num_instances = max(int(x.max()) for x in panoptic_list_train)+1
    elif method == 'gt':
        if data_source == 'dslr':
            files = glob.glob(f'/fs/nexus-projects/3D_r2s/dataset/scannetpp_panoptica_full_dslr/{scene_name}/panoptic/*')
        elif data_source == 'iphone':
            files = sorted(glob.glob(f'/fs/nexus-projects/3D_r2s/dataset/scannetpp_panoptica_full/{scene_name}/panoptic/*'))
        num_instances = 0
        for fn in sorted(files):
            objects = Image.open(fn)
            objects = np.asarray(objects)[:,:,1]
            num_instances = max(num_instances, objects.max())
        num_instances = num_instances+1

    reading_dir = "images" if images == None else images
    if data_source == 'dslr':
        if not free_viewpoint:
            test_cam_infos = []
        else:
            test_cam_infos = readScannetCamerasFreeViewpoint(path, scene_name, images_folder=os.path.join(path, reading_dir))
    elif data_source == 'iphone':
        assert not free_viewpoint, "Free viewpoint not supported for iPhone data"
        root_dir = Path(f'/fs/nexus-projects/3D_r2s/dataset/Scannetpp/NVS/data/{scene_name}/iphone')
        candidates = []
        for idx, fn in enumerate(sorted([x.stem for x in (root_dir / "rgb").iterdir() if x.name.endswith('.jpg')], key=lambda y: int(y) if y.isnumeric() else y)):
            if idx % 10 == 0:
                continue
            if not os.path.exists(os.path.join(str(root_dir).replace('Scannetpp/NVS/data', 'scannetpp_panoptica_full').replace('iphone', 'panoptic'), fn + '.png')):
                continue
            candidates.append(idx)
        random.shuffle(candidates)
        eval_list = candidates[:50]
        test_cam_infos = readScannetCameras(path, panoptic_list_train, images_folder=os.path.join(path, reading_dir), method=method, data_source=data_source, eval_list=eval_list)
    train_cam_infos = readScannetCameras(path, panoptic_list_train, images_folder=os.path.join(path, reading_dir), method=method, data_source=data_source)
    # pdb.set_trace()

    nerf_normalization = getNerfppNorm(train_cam_infos)
    mesh = trimesh.load(f'/fs/nexus-projects/3D_r2s/dataset/Scannetpp/NVS/data/{scene_name}/scans/mesh_aligned_0.05.ply')
    xyz = np.array(mesh.vertices)
    if data_source == 'dslr':
        Rot = np.array([
            [0, 1,  0],
            [1, 0,  0],
            [0, 0, -1],
        ])
        # For the DSLR data, we need to rotate the point cloud to match the camera coordinate system
        xyz = xyz@Rot.T
    rgb = np.array(mesh.visual.vertex_colors)[:,:3]
    num_sample = 200000
    N = xyz.shape[0]
    if N > num_sample:
        idx = np.random.choice(N, size=num_sample, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    ply_path = os.path.join(path, "colmap/points3D_mesh.ply")
    storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           num_classes=num_instances)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}