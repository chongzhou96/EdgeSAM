import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import random
import math
from kornia.contrib import distance_transform
from typing import List


def cal_iou(a, b):
    intersect = ((a > 0.5) & (b > 0.5)).sum(dim=(2, 3))
    union = ((a > 0.5) | (b > 0.5)).sum(dim=(2, 3))
    return intersect / union


def make_overlap(gt, pred):
    h, w = gt.shape
    show = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0)]

    tp = (gt == pred) * (gt == 1)
    fp = (gt != pred) * (gt == 0)
    fn = (gt != pred) * (gt == 1)

    show[tp] = colors[0]
    show[fp] = colors[1]
    show[fn] = colors[2]
    return show


def draw_point_on_figure(fig, point, label, marker_size=1):
    for p, l in zip(point, label):
        if l == 1:  # positive
            fig.plot(p[0], p[1], marker='o', color='yellow', markersize=marker_size)
        elif l == 0:  # negative
            fig.plot(p[0], p[1], marker='o', color='blue', markersize=marker_size)


def make_fig(img, gt, pred, point_prompt=None, box_prompt=None, mask_prompt=None, marker_size=10):
    img = img.permute(1, 2, 0).cpu().numpy()
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    if point_prompt is not None:
        point = point_prompt[0].cpu()
        label = point_prompt[1].cpu()
        num_prompt = point.shape[0]
    if box_prompt is not None:
        num_prompt = box_prompt.shape[0]

    fig_num = 4
    fig_size = (2, 2)  # w, h
    fig = plt.figure(figsize=(fig_num * fig_size[0], num_prompt * fig_size[1]))
    count = 1
    for i in range(num_prompt):
        fig.add_subplot(num_prompt, fig_num, count, xticks=[], yticks=[], title='img')
        count += 1
        img_with_prompt = img.copy()
        if box_prompt is not None:
            top_left, bottom_right = box_prompt[i, :2].int().tolist(), box_prompt[i, 2:].int().tolist()
            cv2.rectangle(img_with_prompt, top_left, bottom_right, (0, 255, 0), marker_size)
        plt.imshow(img_with_prompt)
        if point_prompt is not None:
            draw_point_on_figure(plt, point[i], label[i], marker_size // 2)

        fig.add_subplot(num_prompt, fig_num, count, xticks=[], yticks=[], title='gt')
        count += 1
        plt.imshow(gt[i])

        fig.add_subplot(num_prompt, fig_num, count, xticks=[], yticks=[], title='pred')
        count += 1
        plt.imshow(pred[i])

        fig.add_subplot(num_prompt, fig_num, count, xticks=[], yticks=[], title='overlap')
        count += 1
        overlap = make_overlap(gt[i], pred[i])
        plt.imshow(overlap)
        if point_prompt is not None:
            draw_point_on_figure(plt, point[i], label[i], marker_size // 2)

    return fig


def xywh2xyxy(xywh):
    top_left = xywh[:, :2]
    bottom_right = xywh[:, :2] + xywh[:, 2:]
    return torch.cat((top_left, bottom_right), dim=1)


def xyxy2xywh(xyxy):
    top_left = xyxy[:, :2]
    width = xyxy[:, 2] - xyxy[:, 0]
    height = xyxy[:, 3] - xyxy[:, 1]
    return torch.cat((top_left, width[:, None], height[:, None]), dim=1)


def sample_point_in_mask(mask, gt, num_samples=1, slic_labels=None):
    if len(mask.shape) == 4:
        mask = mask[:, 0]
        gt = gt[:, 0]

    device = mask.device
    sample_list = []
    label_list = []
    fp = (mask != gt) * (gt == 0)
    fn = (mask != gt) * (gt == 1)
    fp_fn = fp | fn

    label_map = -2 * torch.ones_like(mask, dtype=torch.int32)
    label_map[fp] = 0
    label_map[fn] = 1

    _, h, w = mask.shape
    y_axis = torch.arange(h, device=device)[:, None].expand(h, w)
    x_axis = torch.arange(w, device=device)[None, :].expand(h, w)
    grid_points = torch.stack([x_axis, y_axis], dim=-1)

    if slic_labels is not None:
        slic_labels = torch.from_numpy(slic_labels).to(device)

    # TODO parallelize
    for cur_fp_fn, cur_label_map in zip(fp_fn, label_map):
        h, w = cur_fp_fn.shape
        if slic_labels is not None:
            cur_slic_labels = slic_labels.clone()
            cur_slic_labels[~cur_fp_fn] = -1
            unique, counts = torch.unique(cur_slic_labels, return_counts=True)
            # ignore the region with label -1 (the first item)
            unique = unique[1:]
            counts = counts[1:]

            # selected the largest SLIC region
            # if num_samples == 1:
            #     u = unique[counts.argmax()]
            #     c = counts.max()
            #     keep_one = torch.randint(c, (1,))
            #     sample_list.append(grid_points[cur_slic_labels == u][keep_one])
            #     label_list.append(cur_label_map[cur_slic_labels == u][keep_one])
            #     continue

            freq = (counts / counts.sum()).tolist()
            candidate_points, candidate_labels = [], []
            for u, c in zip(unique, counts):
                # only keep one pixel in each super pixel group
                keep_one = torch.randint(c, (1,))
                candidate_points.append(grid_points[cur_slic_labels == u][keep_one])
                candidate_labels.append(cur_label_map[cur_slic_labels == u][keep_one])
            if len(candidate_points) < num_samples:
                sample_list.append(torch.zeros(num_samples, 2, device=device))
                label_list.append(-torch.ones(num_samples, device=device) * 2)  # to ignore
            else:
                selected = random.choices(range(len(candidate_points)), freq, k=num_samples)
                sample_list.append(torch.cat([candidate_points[i] for i in selected], dim=0))
                label_list.append(torch.cat([candidate_labels[i] for i in selected], dim=0))
        else:
            # TODO update the criteria for aborting sampling
            if cur_fp_fn.sum() < num_samples * 10:
                sample_list.append(torch.zeros(num_samples, 2, device=device))
                label_list.append(-2 * torch.ones(num_samples, device=device))  # to ignore
            else:
                candidate_points = grid_points[cur_fp_fn]
                candidate_labels = cur_label_map[cur_fp_fn]
                selected = torch.randint(candidate_points.shape[0], (num_samples,))
                sample_list.append(candidate_points[selected])
                label_list.append(candidate_labels[selected])
    return torch.stack(sample_list, dim=0), torch.stack(label_list, dim=0)


def get_img_bs(points=None, boxes=None, masks=None):
    if points is not None:
        return len(points)
    elif boxes is not None:
        return len(boxes)
    elif masks is not None:
        return len(masks)
    else:
        raise ValueError('At least one type of prompts needs to be provided.')


def get_smallest_prompt_bs(points=None, boxes=None, masks=None):
    res = math.inf
    if points is not None:
        for item in points:
            if item.size(0) < res:
                res = item.size(0)
        return res
    elif boxes is not None:
        for item in boxes:
            if item.size(0) < res:
                res = item.size(0)
        return res
    elif masks is not None:
        for item in masks:
            if item.size(0) < res:
                res = item.size(0)
        return res
    else:
        raise ValueError('At least one type of prompts needs to be provided.')


def sample_prompts(points, boxes, masks, max_allowed_prompts, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    img_bs = get_img_bs(points, boxes, masks)
    sample_points, sample_boxes, sample_masks = [], [], []
    for i in range(img_bs):
        if points is not None:
            num_prompts = points[i].size(0)
        elif boxes is not None:
            num_prompts = boxes[i].size(0)
        elif masks is not None:
            num_prompts = masks[i].size(0)
        else:
            raise ValueError('points, boxes, masks cannot all be None at the same time.')

        if num_prompts > max_allowed_prompts > 0:
            selected = torch.randint(0, num_prompts, (max_allowed_prompts,))
            if points is not None:
                sample_points.append(points[i][selected])
            if boxes is not None:
                sample_boxes.append(boxes[i][selected])
            if masks is not None:
                sample_masks.append(masks[i][selected])
        else:
            if points is not None:
                sample_points.append(points[i])
            if boxes is not None:
                sample_boxes.append(boxes[i])
            if masks is not None:
                sample_masks.append(masks[i])

    if points is None: sample_points = None
    if boxes is None: sample_boxes = None
    if masks is None: sample_masks = None

    return sample_points, sample_boxes, sample_masks


def get_centroid_from_mask(masks):
    assert len(masks.shape) == 4
    centroids = []
    for mask in masks:
        mask = mask[None, :]
        w = mask.shape[3]
        mask_dt = distance_transform((~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:, :, 1:-1,
                  1:-1]
        centroid = torch.tensor([mask_dt.argmax() / w, mask_dt.argmax() % w], device=mask.device).long().flip(0)
        centroids.append(centroid)
    return torch.stack(centroids, dim=0)[:, None]


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords