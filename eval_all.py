# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
from typing import Any, Dict, List, Set, Tuple
import pdb
import copy, json

def _is_set_int(value: Any) -> bool:
    """Check wheter value is a `Set[int]`"""
    return isinstance(value, Set) and set(map(type, value)).issubset({int})

def _validate_categories(things: Set[int], stuff: Set[int]) -> None:
    """Validate metric arguments `things` and `stuff`."""
    if not _is_set_int(things):
        raise ValueError("Expected argument `things` to be of type `Set[int]`")
    if not _is_set_int(stuff):
        raise ValueError("Expected argument `stuff` to be of type `Set[int]`")
    if stuff & things:
        raise ValueError("Expected arguments `things` and `stuffs` to have distinct keys.")


def _validate_inputs(preds: torch.Tensor, target: torch.Tensor) -> None:
    """Validate predictions and target have the correct shape."""
    if not isinstance(preds, torch.Tensor):
        raise ValueError("Expected argument `preds` to be of type `torch.Tensor`")
    if not isinstance(target, torch.Tensor):
        raise ValueError("Expected argument `target` to be of type `torch.Tensor`")
    if preds.shape != target.shape:
        raise ValueError("Expected argument `preds` and `target` to have the same shape")

def _get_void_color(things: Set[int], stuff: Set[int]) -> Tuple[int, int]:
    unused_category_id = 1 + max([0] + list(things) + list(stuff))
    return unused_category_id, 0

def _get_category_id_to_continous_id(things: Set[int], stuff: Set[int]) -> Dict[int, int]:
    # things metrics are stored with a continous id in [0, len(things)[,
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    # stuff metrics are stored with a continous id in [len(things), len(things) + len(stuffs)[
    stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuff)}
    cat_id_to_continuous_id = {}
    cat_id_to_continuous_id.update(thing_id_to_continuous_id)
    cat_id_to_continuous_id.update(stuff_id_to_continuous_id)
    return cat_id_to_continuous_id

def get_non_robust_classes_for_image(pred_sem, target_sem, robustness_thres=0.005):
    pred_unique, pred_counts = pred_sem.unique(return_counts=True)
    target_unique, target_counts = target_sem.unique(return_counts=True)
    pred_perc = pred_counts / pred_counts.sum()
    target_perc = target_counts / target_counts.sum()
    return set(pred_unique[pred_perc < robustness_thres].tolist() + target_unique[target_perc < robustness_thres].tolist())

def _isin(arr: torch.tensor, values: List) -> torch.Tensor:
    """basic implementation of torch.isin to support pre 0.10 version."""
    return (arr[..., None] == arr.new(values)).any(-1)

def _prepocess_image(
    things: Set[int],
    stuff: Set[int],
    img: torch.Tensor,
    void_color: Tuple[int, int],
    allow_unknown_category: bool,
) -> torch.Tensor:  # flatten the height*width dimensions
    img = torch.flatten(img, 0, -2)
    stuff_pixels = _isin(img[:, 0], list(stuff))
    things_pixels = _isin(img[:, 0], list(things))
    # reset instance ids of stuffs
    img[stuff_pixels, 1] = 0
    if not allow_unknown_category and not torch.all(things_pixels | stuff_pixels):
        raise ValueError("Unknown categories found in preds")
    # set unknown categories to void color
    img[~(things_pixels | stuff_pixels)] = img.new(void_color)
    return img

def _nested_tuple(nested_list: List) -> Tuple:
    """Construct a nested tuple from a nested list."""
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list

def _totuple(t: torch.Tensor) -> Tuple:
    """Convert a tensor into a nested tuple."""
    return _nested_tuple(t.tolist())

def _get_color_areas(img: torch.Tensor) -> Dict[Tuple, torch.Tensor]:
    """Calculate a dictionary {pixel_color: area}."""
    unique_keys, unique_keys_area = torch.unique(img, dim=0, return_counts=True)
    # dictionary indexed by color tuples
    return dict(zip(_totuple(unique_keys), unique_keys_area))

def _panoptic_quality_update(
    flatten_preds: torch.Tensor,
    flatten_target: torch.Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Returns stat scores (iou sum, true positives, false positives, false negatives) required
    to compute accuracy.
    """
    device = flatten_preds.device
    n_categories = len(cat_id_to_continuous_id)
    iou_sum = torch.zeros(n_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(n_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(n_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(n_categories, dtype=torch.int, device=device)

    # calculate the area of each prediction, ground truth and pairwise intersection
    pred_areas = _get_color_areas(flatten_preds)
    target_areas = _get_color_areas(flatten_target)
    # intersection matrix of shape [height, width, 2, 2]
    intersection_matrix = torch.transpose(torch.stack((flatten_preds, flatten_target), -1), -1, -2)
    intersection_areas = _get_color_areas(intersection_matrix)

    # select intersection of things of same category with iou > 0.5
    pred_segment_matched = set()
    target_segment_matched = set()
    for (pred_color, target_color), intersection in intersection_areas.items():
        # test only non void, matching category
        if target_color == void_color:
            continue
        if pred_color[0] != target_color[0]:
            continue
        continuous_id = cat_id_to_continuous_id[pred_color[0]]
        pred_area = pred_areas[pred_color]
        target_area = target_areas[target_color]
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        union = pred_area - pred_void_area + target_area - void_target_area - intersection
        iou = intersection / union

        if iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            iou_sum[continuous_id] += iou
            true_positives[continuous_id] += 1

    # count false negative: ground truth but not matched
    # areas that are mostly void in the prediction are ignored
    false_negative_colors = set(target_areas.keys()).difference(target_segment_matched)
    false_negative_colors.discard(void_color)
    for target_color in false_negative_colors:
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        if void_target_area / target_areas[target_color] > 0.5:
            continue
        continuous_id = cat_id_to_continuous_id[target_color[0]]
        false_negatives[continuous_id] += 1

    # count false positive: predicted but not matched
    # areas that are mostly void in the target are ignored
    false_positive_colors = set(pred_areas.keys()).difference(pred_segment_matched)
    false_positive_colors.discard(void_color)
    for pred_color in false_positive_colors:
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        if pred_void_area / pred_areas[pred_color] > 0.5:
            continue
        continuous_id = cat_id_to_continuous_id[pred_color[0]]
        false_positives[continuous_id] += 1

    return iou_sum, true_positives, false_positives, false_negatives

def _panoptic_quality_compute(
    things: Set[int],
    stuff: Set[int],
    iou_sum: torch.Tensor,
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
) -> Dict:
    # TODO: exclude from mean categories that are never seen ?
    # TODO: per class metrics

    # per category calculation
    denominator = (true_positives + 0.5 * false_positives + 0.5 * false_negatives).double()
    panoptic_quality = torch.where(denominator > 0.0, iou_sum / denominator, 0.0)
    segmentation_quality = torch.where(true_positives > 0.0, iou_sum / true_positives, 0.0)
    recognition_quality = torch.where(denominator > 0.0, true_positives / denominator, 0.0)

    metrics = dict(
        all=dict(
            pq=torch.mean(panoptic_quality),
            rq=torch.mean(recognition_quality),
            sq=torch.mean(segmentation_quality),
            n=len(things) + len(stuff),
        ),
        things=dict(
            pq=torch.mean(panoptic_quality[: len(things)]),
            rq=torch.mean(recognition_quality[: len(things)]),
            sq=torch.mean(segmentation_quality[: len(things)]),
            n=len(things),
        ),
        stuff=dict(
            pq=torch.mean(panoptic_quality[len(things) :]),
            rq=torch.mean(recognition_quality[len(things) :]),
            sq=torch.mean(segmentation_quality[len(things) :]),
            n=len(stuff),
        ),
    )
    return metrics

def panoptic_quality(
    preds: torch.Tensor,
    target: torch.Tensor,
    things: Set[int],
    stuff: Set[int],
    allow_unknown_preds_category: bool = False,
    robust: float = 0.005
) -> Tuple[Any, Any, Any]:
    unused_classes = things.union(stuff) - set(preds[..., 0].unique().tolist() + target[..., 0].unique().tolist())
    non_robust_classes = get_non_robust_classes_for_image(preds[..., 0], target[..., 0], robust)
    things = things - unused_classes - non_robust_classes
    stuff = stuff - unused_classes - non_robust_classes
    _validate_categories(things, stuff)
    _validate_inputs(preds, target)
    void_color = _get_void_color(things, stuff)
    cat_id_to_continuous_id = _get_category_id_to_continous_id(things, stuff)
    flatten_preds = _prepocess_image(things, stuff, preds, void_color, allow_unknown_preds_category)
    flatten_target = _prepocess_image(things, stuff, target, void_color, True)
    iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
        flatten_preds, flatten_target, cat_id_to_continuous_id, void_color
    )
    results = _panoptic_quality_compute(things, stuff, iou_sum, true_positives, false_positives, false_negatives)
    return results["all"]["pq"], results["all"]["sq"], results["all"]["rq"]

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=10000):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.int32)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

# def visualize_obj(objects):
#     rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.int32)
#     all_obj_ids = np.unique(objects)
#     for id in all_obj_ids:
#         colored_mask = id2rgb(id)
#         rgb_mask[objects == id] = colored_mask
#     return rgb_mask

def visualize_obj(objects, max_num_obj=10000):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(int(id), max_num_obj=max_num_obj)
        rgb_mask[objects == id] = colored_mask

    return rgb_mask


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    pqs = []
    sqs = []
    rqs = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.int32))

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.int32))
        # valid = gt_objects != 0
        
        # pano_pred = torch.cat([
        #     torch.ones_like(pred_obj[valid]).unsqueeze(1),
        #     pred_obj[valid].unsqueeze(1),
        # ], dim=1)

        # pano_target = torch.cat([
        #     torch.ones_like(gt_objects[valid]).unsqueeze(1),
        #     gt_objects[valid].unsqueeze(1),
        # ], dim=1)

        # pred_obj, gt_objects: H x W
        # 0 is background / unlabeled / void

        pano_pred = torch.stack([
            torch.ones_like(pred_obj),
            pred_obj,
        ], dim=-1)

        pano_target = torch.stack([
            torch.ones_like(gt_objects),
            gt_objects,
        ], dim=-1)

        bg_t = gt_objects == 0
        pano_target[bg_t] = torch.tensor([0, 0], device=pano_target.device, dtype=pano_target.dtype)

        bg_p = pred_obj == 0
        pano_pred[bg_p] = torch.tensor([0, 0], device=pano_pred.device, dtype=pano_pred.dtype)
        # print(gt_objects.max(), pred_obj.max())

        # pdb.set_trace()
        metric_pq, metric_sq, metric_rq = panoptic_quality(
            pano_pred,
            pano_target,
            things={1},
            stuff=set(),
            allow_unknown_preds_category=True,
            robust=0.0
        )
        pqs.append(metric_pq.item())
        sqs.append(metric_sq.item())
        rqs.append(metric_rq.item())
        # pdb.set_trace()

        rgb_mask = feature_to_rgb(rendering_obj)
        # Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        # Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        # Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    print("Average PQ: ", np.mean(pqs))
    print("Average SQ: ", np.mean(sqs))
    print("Average RQ: ", np.mean(rqs))
    pq, sq, rq = np.mean(pqs), np.mean(sqs), np.mean(rqs)
    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    size = (gt.shape[-1]*5,gt.shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    # writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path,file_name)))
        rgb = np.array(Image.open(os.path.join(render_path,file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

        result = np.hstack([gt,rgb,gt_obj,pred_obj,render_obj])
        result = result.astype('uint8')

        # Image.fromarray(result).save(os.path.join(out_path,file_name))
        # writer.write(result[:,:,::-1])

    # writer.release()
    return pq, sq, rq


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            pq, sq, rq = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)
            return pq, sq, rq
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # pdb.set_trace()

    scene_names = [
        "0d2ee665be",
        "1ada7a0617",
        "3e8bba0176",
        "3f15a9266d",
        "5ee7c22ba0",
        "7bc286c1b6",
        "a24f64f7fb",
        "5748ce6f01",
        "f9f95681fd",
        "3864514494"
    ]
    with open('./num_instances.json', 'r') as f:
        num_instances_dict = json.load(f)
    pqs = []
    sqs = []
    rqs = []
    for scene_name in scene_names:
        print(scene_name)
        args_ = copy.deepcopy(args)
        args_.source_path = args_.source_path.replace('1ada7a0617', scene_name)
        args_.num_classes = num_instances_dict[scene_name]
        args_.model_path = args_.model_path.replace('1ada7a0617', scene_name)

        pq, sq, rq = render_sets(model.extract(args_), args_.iteration, pipeline.extract(args_), args_.skip_train, args_.skip_test)
        pqs.append(pq)
        sqs.append(sq)
        rqs.append(rq)
    
    print("Average PQ: ", np.mean(pqs))
    print("Average SQ: ", np.mean(sqs))
    print("Average RQ: ", np.mean(rqs))