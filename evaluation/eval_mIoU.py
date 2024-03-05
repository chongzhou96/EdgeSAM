import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from mmengine.dataset import pseudo_collate, worker_init_fn, DefaultSampler
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from mmengine.utils import ProgressBar
from mmdet.structures.bbox import bbox_overlaps
import argparse
import os
import random

from edge_sam import build_sam_from_config
from training.data import SA1BDataset, COCODataset
from edge_sam.utils.common import (cal_iou, sample_point_in_mask, xywh2xyxy, xyxy2xywh,
                                           make_fig, sample_prompts, get_centroid_from_mask)
from edge_sam.config import _C, _update_config_from_file
from functools import partial

parser = argparse.ArgumentParser(
    description=(
        "Evaluate SAM models on the SA-1B / COCO dataset"
    )
)

parser.add_argument(
    "sam_cfg_file",
    type=str,
    help="The path to the sam config file",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="sa",
    choices=["coco", "sa", "cocofied_lvis", "lvis"],
    help="Evaluation dataset, options: sa, coco, cocofied_lvis, lvis",
)

parser.add_argument(
    "--out-name",
    type=str,
    default="",
    help="The name of the output record file",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="weights/sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--max-prompt-bs",
    type=int,
    default=64,
    help="Max number of prompts allowed for a single image (avoid OOM).",
)

parser.add_argument(
    "--num-samples",
    type=int,
    default=-1,
    help="Number of samples in the subset of the current dataset. If less than 0, then sample the whole dataset",
)

parser.add_argument(
    "--sort",
    action="store_true",
)

parser.add_argument(
    "--save-json",
    action="store_true",
)

parser.add_argument(
    "--refine-iter",
    type=int,
    default=1,
    help="Refine the mask decoder by how many iteration. If 0, no refinement will be performed"
)

parser.add_argument(
    "--img-bs",
    type=int,
    default=1,
    help="Image batch size."
)

parser.add_argument(
    "--num-multimask-outputs",
    type=int,
    default=1,
    choices=[1, 3, 4],
    help="The number of mask output for each prompt."
)

parser.add_argument(
    "--multimask-select",
    type=str,
    default='score',
    choices=['score', 'area', 'oracle'],
    help="The criteria to select one mask per prompt."
)

parser.add_argument(
    "--prompt-types",
    type=str,
    default=['box'],
    nargs='+',
    help="Prompt types"
)

parser.add_argument(
    "--point-from",
    type=str,
    default="point",
    choices=["mask-rand", "mask-center", "box-center", "point"],
    help="Evaluation dataset, options: sam, coco",
)

parser.add_argument(
    "--vis",
    action="store_true",
    help="Enable tensorboard visualization"
)

parser.add_argument(
    "--slic",
    action="store_true",
    help="Use SLIC superpixel segmentation to provide guidance for point sampling"
)

parser.add_argument(
    "--rpn",
    type=str,
    help="Use RPN to extract proposals that will be used as prompts"
)

parser.add_argument(
    "--mask-output-dir",
    type=str,
    help="Directory path to save the output masks."
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)


def point_in_boxes(point, boxes, scores):
    selected = (boxes[:, 0] < point[0]) & \
               (point[0] < boxes[:, 2]) & \
               (boxes[:, 1] < point[1]) & \
               (point[1] < boxes[:, 3])
    return boxes[selected], scores[selected]


def fast_nms(boxes, scores, iou_thr=0.5):
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx, :]

    iou = bbox_overlaps(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=0)
    keep = iou_max <= iou_thr
    return boxes[keep], scores[keep]


def point_box_dist(point, boxes):
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    center = torch.stack([center_x, center_y], dim=1)
    point = point[None].expand(boxes.shape[0], 2)
    return F.pairwise_distance(point, center)


def k_nearest_center(point, boxes, scores, topk=1):
    distance = -point_box_dist(point, boxes)
    selected = distance.topk(min(topk, boxes.shape[0]), dim=0)[1]
    return boxes[selected], scores[selected]


@torch.no_grad()
def main(args):
    if args.launcher == 'none':
        _distributed = False
    else:
        _distributed = True

    if _distributed and not is_distributed():
        init_dist(args.launcher)

    config = _C.clone()
    _update_config_from_file(config, args.sam_cfg_file)
    config.defrost()
    config.DISTILL.ENCODER_ONLY = False
    if args.rpn:
        config.DISTILL.RPN_HEAD = args.rpn
    config.freeze()

    use_rpn = args.rpn is not None

    # TODO change the hard-coded out_indices
    sam = build_sam_from_config(
        config, enable_batch=True,
        use_rpn=use_rpn, out_indices=(2, 3, 4, 5))

    with open(args.checkpoint, "rb") as f:
        state_dict = torch.load(f)
    print(sam.load_state_dict(state_dict, strict=False))

    sam.to(device=args.device)
    sam.eval()

    if args.dataset == 'sa':
        dataset = SA1BDataset(data_root='datasets/SA-1B', split='val', num_samples=args.num_samples,
                               filter_by_area=None, sort_by_area=args.sort, load_gt_mask=True,
                               max_allowed_prompts=args.max_prompt_bs, fix_seed=True)
        marker_size = 10
    elif args.dataset in ['coco', 'cocofied_lvis', 'lvis']:
        dataset = COCODataset(data_root='datasets/coco', split='val', num_samples=args.num_samples,
                              filter_by_area=None, sort_by_area=args.sort, load_gt_mask=True,
                              max_allowed_prompts=args.max_prompt_bs, fix_seed=True, annotation=args.dataset)
        marker_size = 5
    else:
        raise NotImplemented
    sampler = DefaultSampler(dataset, False)
    init_fn = partial(
        worker_init_fn,
        num_workers=1,
        rank=get_rank(),
        seed=0,
        disable_subprocess_warning=True)
    dataloader = DataLoader(
        dataset, batch_size=args.img_bs, sampler=sampler, drop_last=False,
        collate_fn=pseudo_collate, worker_init_fn=init_fn)

    if get_rank() == 0:
        progress_bar = ProgressBar(len(dataloader))

    model_type = config.MODEL.TYPE
    writer_root = 'output/vis/'
    postfix = f'{model_type}'
    if config.DISTILL.FUSE:
        postfix += '_fuse'
    if args.rpn:
        postfix += f'_{args.rpn}'
    postfix += f'_{args.max_prompt_bs}prompts'
    if args.sort:
        postfix += '_sort'
    if args.slic:
        postfix += '_slic'
    postfix += f'_{args.dataset}'

    writer_path = writer_root + postfix
    if args.vis:
        writer = SummaryWriter(writer_path)

    iou_point = [[] for _ in range(args.refine_iter)]
    torch.manual_seed(0)

    for imgs, annos in dataloader:
        imgs = torch.stack(imgs, dim=0).cuda(non_blocking=True)
        img_size_before_pad = annos['img_size_before_pad']
        img_size_pad = (sam.image_encoder.img_size, sam.image_encoder.img_size)
        mask_threshold = sam.mask_threshold

        image_encoder_outs = sam.image_encoder(imgs)
        if use_rpn:
            image_embeddings = image_encoder_outs[-1]
        else:
            image_embeddings = image_encoder_outs

        dense_pe = sam.prompt_encoder.get_dense_pe()

        if 'prompt_point' in annos:
            points = annos['prompt_point']
            points = torch.cat(points, dim=0)
            points = points.cuda(non_blocking=True)
            labels = torch.ones(points.shape[:2], device=imgs.device)
            points = (points, labels)
        else:
            points = None

        boxes = annos['prompt_box']
        num_prompts = []
        for box in boxes:
            num_prompts.append(box.size(0))

        boxes = torch.cat(boxes, dim=0)
        boxes = boxes.cuda(non_blocking=True)

        gt_mask = annos['gt_mask']
        gt_mask = torch.cat(gt_mask, dim=0)
        gt_mask = gt_mask.float().cuda(non_blocking=True)[:, None]

        if args.point_from == 'mask-rand':
            point_list = []
            for g in gt_mask.squeeze(1):
                candidate_indices = g.nonzero()
                selected_index = random.randint(0, len(candidate_indices) - 1)
                p = candidate_indices[selected_index].flip(0)
                point_list.append(p)
            points = torch.stack(point_list, dim=0)[:, None]
            labels = torch.ones(points.shape[:2], device=imgs.device)
            points = (points, labels)
        elif args.point_from == 'box-center':
            center_x = (boxes[:, 0] + boxes[:, 1]) / 2
            center_y = (boxes[:, 2] + boxes[:, 3]) / 2
            boxes = None
            points = torch.stack([center_x, center_y], dim=1)[:, None]
            labels = torch.ones(points.shape[:2], device=imgs.device)
            points = (points, labels)
        elif args.point_from == 'mask-center':
            points = get_centroid_from_mask(gt_mask > 0.5)
            labels = torch.ones(points.shape[:2], device=imgs.device)
            points = (points, labels)

        if 'point' not in args.prompt_types:
            points = None
        if 'box' not in args.prompt_types:
            boxes = None

        box_labels = None
        if points is not None and use_rpn:
            proposals = sam.forward_rpn(image_encoder_outs[:-1], score_thr=0.1, with_nms=False)
            selected_boxes_list = []
            box_labels_list = []
            coords = points[0].squeeze(1)
            for img_i, num_prompt in enumerate(num_prompts):
                p_boxes = proposals[img_i]['bboxes']
                p_scores = proposals[img_i]['scores']
                box_labels = torch.ones(num_prompt).cuda()

                # p_boxes, p_scores = fast_nms(
                #     p_boxes, p_scores, iou_thr=0.5)

                for prompt_i in range(num_prompt):
                    cur_point = coords[prompt_i]

                    selected_boxes = p_boxes
                    selected_scores = p_scores

                    selected_boxes, selected_scores = point_in_boxes(
                        cur_point, selected_boxes, selected_scores)

                    if selected_boxes.size(0) == 0:
                        merged_box = torch.zeros(4).cuda()
                        box_labels[prompt_i] = -1
                    else:
                        selected_boxes, selected_scores = k_nearest_center(
                            cur_point, selected_boxes, selected_scores, topk=5)

                        # box_weight = selected_scores / selected_scores.sum()
                        box_weight = selected_scores >= selected_scores.max()
                        merged_box = (selected_boxes * box_weight[:, None]).sum(dim=0)

                        # if point_box_dist(cur_point, merged_box[None])[0] > 32:
                        #     merged_box = torch.zeros(4).cuda()
                        #     box_labels[prompt_i] = -1

                        # if merged_box[3]-merged_box[1] < 32 or merged_box[2]-merged_box[0] < 32:
                        #     merged_box = torch.zeros(4).cuda()
                        #     box_labels[prompt_i] = -1

                    selected_boxes_list.append(merged_box)
                box_labels_list.append(box_labels)
            boxes = torch.stack(selected_boxes_list, dim=0)
            box_labels = torch.cat(box_labels_list, dim=0)

        img_bs = imgs.size(0)
        valid = torch.zeros(img_bs, 1, *img_size_pad, device=imgs.device)
        valid_list = []
        for img_i in range(img_bs):
            h, w = img_size_before_pad[img_i][1:]
            valid[img_i, :, :h, :w] = 1
            valid_list.append(valid[img_i:img_i + 1].expand(num_prompts[img_i], *valid.shape[1:]))
        valid = torch.cat(valid_list, dim=0)

        slic_labels = None
        # if args.slic:
        #     # slic_labels = slic(img.cpu().numpy(), start_label=0, channel_axis=0)
        #     img_numpy = imgs.permute(1, 2, 0).contiguous().cpu().numpy()
        #     slic_labels = SlicAvx2(num_components=100, compactness=10).iterate(img_numpy)

        for iter_i in range(args.refine_iter):
            prev_point = points

            if iter_i > 0:
                valid_down = F.interpolate(valid, mask_pred.shape[2:], mode="bilinear", align_corners=False)
                gt_mask_down = F.interpolate(gt_mask, mask_pred.shape[2:], mode="bilinear", align_corners=False)

                mask_pred_valid = (mask_pred > mask_threshold) * valid_down
                gt_mask_valid = (gt_mask_down > mask_threshold) * valid_down

                point, label = sample_point_in_mask(mask_pred_valid, gt_mask_valid, num_samples=1,
                                                    slic_labels=slic_labels)

                point[..., 0] = point[..., 0] / mask_pred.shape[3] * img_size_pad[1]
                point[..., 1] = point[..., 1] / mask_pred.shape[2] * img_size_pad[0]

                if prev_point is not None:
                    point = torch.cat([prev_point[0], point], dim=1)
                    label = torch.cat([prev_point[1], label], dim=1)
                points = (point, label)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=points, boxes=boxes, masks=None,
                num_prompts=num_prompts, box_labels=box_labels)

            mask_pred, iou_pred = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                num_multimask_outputs=args.num_multimask_outputs,
                num_prompts=num_prompts
            )

            if args.num_multimask_outputs > 1:
                n, c, h, w = mask_pred.shape
                if args.multimask_select == 'score':
                    max_score_idx = iou_pred.argmax(dim=1)[:, None, None, None].expand(n, 1, h, w)
                    # mask_pred = mask_pred[:, max_score_idx]
                    mask_pred = torch.gather(mask_pred, 1, max_score_idx)
                elif args.multimask_select == 'area':
                    area = (mask_pred > mask_threshold).sum(dim=(2, 3))
                    max_area_idx = area.argmax(dim=1)[:, None, None, None].expand(n, 1, h, w)
                    mask_pred = torch.gather(mask_pred, 1, max_area_idx)

            # post processing
            mask_pred_up = F.interpolate(mask_pred, img_size_pad, mode="bilinear", align_corners=False)
            for img_i in range(img_bs):
                if img_i == 0:
                    cur = slice(0, num_prompts[img_i])
                else:
                    cur = slice(sum(num_prompts[:img_i]), sum(num_prompts[:img_i + 1]))

                cur_img = imgs[img_i:img_i + 1]
                cur_mask = mask_pred_up[cur]
                cur_gt_mask = gt_mask[cur]
                file_name = annos['info']['file_name'][img_i]
                file_name = file_name.split('.')[0]

                h, w = img_size_before_pad[img_i][1:]
                ori_h, ori_w = annos['info']['height'][img_i], annos['info']['width'][img_i]

                cur_img = cur_img[..., :h, :w]
                cur_mask = cur_mask[..., :h, :w]
                cur_gt_mask = cur_gt_mask[..., :h, :w]

                cur_img = F.interpolate(cur_img, (ori_h, ori_w), mode="bilinear", align_corners=False).squeeze(0)
                cur_mask = F.interpolate(cur_mask, (ori_h, ori_w), mode="bilinear", align_corners=False)
                cur_gt_mask = F.interpolate(cur_gt_mask, (ori_h, ori_w), mode="bilinear", align_corners=False)

                cur_mask = cur_mask > mask_threshold
                cur_gt_mask = cur_gt_mask > mask_threshold

                iou = cal_iou(cur_mask, cur_gt_mask)
                # if args.num_multimask_outputs > 1 and args.multimask_select == 'oracle':
                iou, max_iou_idx = iou.max(dim=1)

                iou_point[iter_i].append(iou.cpu())

                if args.vis:
                    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(args.device)
                    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(args.device)
                    cur_img = (cur_img * pixel_std + pixel_mean).int()

                    # show oracle result
                    n, c, h, w = cur_mask.shape
                    max_iou_idx = max_iou_idx[:, None, None, None].expand(n, 1, h, w)
                    cur_mask = torch.gather(cur_mask, 1, max_iou_idx)
                    cur_gt_mask = cur_gt_mask[:, 0]

                    ori_long_side = max(ori_w, ori_h)

                    cur_box = None
                    if boxes is not None:
                        cur_box = boxes[cur].clone()
                        cur_box = cur_box.reshape(-1, 2, 2)
                        cur_box[..., 0] = cur_box[..., 0] * (ori_long_side / img_size_pad[1])
                        cur_box[..., 1] = cur_box[..., 1] * (ori_long_side / img_size_pad[0])
                        cur_box = cur_box.reshape(-1, 4)

                    cur_point = None
                    if prev_point is not None:
                        cur_coord = prev_point[0][cur].clone()
                        cur_coord[..., 0] = cur_coord[..., 0] * (ori_long_side / img_size_pad[1])
                        cur_coord[..., 1] = cur_coord[..., 1] * (ori_long_side / img_size_pad[0])
                        cur_label = prev_point[1][cur]
                        cur_point = (cur_coord, cur_label)

                    fig = make_fig(cur_img, cur_gt_mask, cur_mask, cur_point, cur_box, marker_size)
                    writer.add_figure(f'{file_name}/{iter_i + 1}', fig)

        if get_rank() == 0:
            progress_bar.update()

    all_iou_point = [[] for _ in range(args.refine_iter)]
    for iter_i in range(args.refine_iter):
        all_iou_point[iter_i] = collect_results(iou_point[iter_i], len(dataset), 'cpu')

    if get_rank() == 0:
        print()
        out_dir = 'output/mIoU'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if len(args.out_name) > 0:
            out_file = args.out_name
        else:
            out_file = args.checkpoint.split('/')[-3]
        f = open(f'{out_dir}/{out_file}', 'a')
        prompt_type = '_'.join(args.prompt_types)
        for iter_i in range(args.refine_iter):
            res = torch.cat(all_iou_point[iter_i], dim=0)
            res = res[~res.isnan()]
            out_string = f'{prompt_type} (iter{iter_i + 1}): {res.mean() * 100:.3f}'
            print(out_string)
            f.write(out_string + '\n')
        f.close()


if __name__ == "__main__":
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    main(args)
