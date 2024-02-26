# --------------------------------------------------------
# EdgeSAM trainign script
# Based on the code: TinyViT
#   (https://github.com/microsoft/Cream/tree/main/TinyViT)
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from my_meter import AverageMeter
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, \
    NativeScalerWithGradNormCount, \
    auto_resume_helper, is_main_process, \
    add_common_args, \
    get_git_info, \
    dice_loss, sigmoid_focal_loss, sigmoid_ce_loss, calculate_uncertainty

from edge_sam import build_sam_from_config, get_config
from edge_sam.utils.common import sample_point_in_mask

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
import loralib

try:
    import wandb
except ImportError:
    wandb = None
NORM_ITER_LEN = 100


VIS = False
if VIS:
    from edge_sam.utils.common import make_fig


def parse_option():
    parser = argparse.ArgumentParser(
        'EdgeSAM training script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(args, config):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_sam_from_config(config, None, True, True)
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    teacher_model = dict()
    if not config.DISTILL.ENCODER_ONLY:
        prompt_encoder_t = copy.deepcopy(model.prompt_encoder)
        for param in prompt_encoder_t.parameters():
            param.requires_grad = False
        prompt_encoder_weights = torch.load('weights/sam_vit_h_prompt_encoder.pth', map_location='cpu')
        msg = prompt_encoder_t.load_state_dict(prompt_encoder_weights, strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded successfully 'prompt_encoder_teacher'")

        mask_decoder_t = copy.deepcopy(model.mask_decoder)
        for param in mask_decoder_t.parameters():
            param.requires_grad = False
        mask_decoder_weights = torch.load('weights/sam_vit_h_mask_decoder.pth', map_location='cpu')
        msg = mask_decoder_t.load_state_dict(mask_decoder_weights, strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded successfully 'mask_decoder_teacher'")

        teacher_model['prompt_encoder'] = prompt_encoder_t
        teacher_model['mask_decoder'] = mask_decoder_t

    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    lr_scheduler = build_scheduler(config, optimizer, len(
        data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        if not config.DISTILL.ENCODER_ONLY:
            load_pretrained(config, model_without_ddp.image_encoder, logger)

            if config.DISTILL.INIT_FROM_TEACHER:
                prompt_encoder_weights = torch.load('weights/sam_vit_h_prompt_encoder.pth', map_location='cpu')
                msg = model_without_ddp.prompt_encoder.load_state_dict(prompt_encoder_weights, strict=False)
                logger.warning(msg)
                logger.info(f"=> loaded successfully 'prompt_encoder'")
                del prompt_encoder_weights

                mask_decoder_weights = torch.load('weights/sam_vit_h_mask_decoder.pth', map_location='cpu')
                msg = model_without_ddp.mask_decoder.load_state_dict(mask_decoder_weights, strict=False)
                logger.warning(msg)
                logger.info(f"=> loaded successfully 'mask_decoder'")
                del mask_decoder_weights
                torch.cuda.empty_cache()

            if config.DISTILL.FREEZE_IMAGE_ENCODER:
                for param in model_without_ddp.image_encoder.parameters():
                    param.requires_grad = False

            if config.DISTILL.FREEZE_PROMPT_ENCODER:
                for param in model_without_ddp.prompt_encoder.parameters():
                    param.requires_grad = False

            if config.DISTILL.FREEZE_MASK_DECODER:
                for param in model_without_ddp.mask_decoder.parameters():
                    param.requires_grad = False
            if config.DISTILL.LORA:
                loralib.mark_only_lora_as_trainable(model_without_ddp.mask_decoder)
        else:
            load_pretrained(config, model_without_ddp, logger)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    loss_writer = None
    if dist.get_rank() == 0:
        loss_writer = SummaryWriter(f'{config.OUTPUT}/{datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")}')

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        teacher_epoch = 0 if config.DISTILL.NO_RAND else epoch
        # set_epoch for dataset_train when distillation
        if hasattr(dataset_train, 'set_epoch'):
            dataset_train.set_epoch(teacher_epoch)
        data_loader_train.sampler.set_epoch(teacher_epoch)

        train_one_epoch_distill_using_saved_embeddings(
            args, config, model, data_loader_train, optimizer, epoch,
            lr_scheduler, loss_scaler, teacher_model, loss_writer)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp,
                            max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def train_one_epoch_distill_using_saved_embeddings(args, config, model, data_loader, optimizer, epoch,
                                                   lr_scheduler, loss_scaler, teacher_model, loss_writer):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    if VIS and dist.get_rank() == 0:
        vis_writer = SummaryWriter('vis')

    for idx, ((samples, annos), (saved_embeddings, seeds)) in enumerate(data_loader):
        samples = torch.stack(samples, dim=0).cuda(non_blocking=True)
        saved_embeddings = torch.as_tensor(np.stack(saved_embeddings, axis=0)).float().cuda(non_blocking=True)

        meters['data_time'].update(time.time() - data_tic)

        img_bs = samples.shape[0]
        img_size_before_pad = annos['img_size_before_pad']

        if not args.only_cpu:
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        if config.DISTILL.ENCODER_ONLY:
            img_size_pad = (model_without_ddp.img_size, model_without_ddp.img_size)
        else:
            img_size_pad = (model_without_ddp.image_encoder.img_size, model_without_ddp.image_encoder.img_size)
            mask_threshold = model_without_ddp.mask_threshold

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            if config.DISTILL.ENCODER_ONLY:
                encoder_embeddings = model(samples)
            else:
                encoder_embeddings = model(mode='image_encoder', x=samples)

        if saved_embeddings.size() != encoder_embeddings.size():
            saved_embeddings = saved_embeddings.reshape(encoder_embeddings.shape)

        loss = dict()
        if config.DISTILL.PIXEL_WISE > 0:
            _tmp = F.mse_loss(encoder_embeddings, saved_embeddings, reduction='none') * config.DISTILL.PIXEL_WISE
            # Get rid of padding with masking
            valid = torch.zeros(img_bs, 1, *img_size_pad, device=samples.device)
            for i in range(img_bs):
                h, w = img_size_before_pad[i][1:]
                valid[i, :, :h, :w] = 1
            valid_downsample = F.interpolate(valid, _tmp.shape[-2:], mode='bilinear', align_corners=False)
            valid_downsample = (valid_downsample > 0.5).flatten(2)
            _tmp = _tmp.flatten(2) * valid_downsample
            _tmp = _tmp.mean(1).sum(-1) / valid_downsample.sum(-1)
            _tmp = _tmp.mean()
            loss['pixel'] = (loss['pixel'] + _tmp) if 'pixel' in loss else _tmp

        # TODO Doesn't make sense to apply KL_DIV on features. Need to support valid_loss.
        if config.DISTILL.CHANNEL_WISE > 0:
            temperature = 4.0
            s = (encoder_embeddings / temperature).flatten(-2, -1).softmax(dim=-1).log()
            t = (saved_embeddings / temperature).flatten(-2, -1).softmax(dim=-1)
            _tmp = F.kl_div(s, t) * config.DISTILL.CHANNEL_WISE * temperature ** 2
            loss['chn'] = (loss['chn'] + _tmp) if 'chn' in loss else _tmp

        # TODO  Need to support valid_loss.
        if config.DISTILL.CORRELATION > 0:
            s = F.normalize(encoder_embeddings, p=2).flatten(-2, -1)
            student_corr = s.transpose(-2, -1) @ s
            t = F.normalize(saved_embeddings, p=2).flatten(-2, -1)
            teacher_corr = t.transpose(-2, -1) @ t
            _tmp = F.mse_loss(student_corr, teacher_corr) * config.DISTILL.CORRELATION
            loss['corr'] = (loss['corr'] + _tmp) if 'corr' in loss else _tmp

        if not config.DISTILL.ENCODER_ONLY:
            dense_pe = model.module.prompt_encoder.get_dense_pe()

            if 'prompt_point' in annos:
                points = annos['prompt_point']
                points = torch.cat(points, dim=0)
                points = points.cuda(non_blocking=True)
                labels = torch.ones(points.shape[:2], device=samples.device)
                points = (points, labels)
            else:
                points = None

            boxes = annos['prompt_box']
            masks = None

            num_prompts = []
            for box in boxes:
                num_prompts.append(box.size(0))

            boxes = torch.cat(boxes, dim=0)
            boxes = boxes.cuda(non_blocking=True)

            if config.DISTILL.PROMPT_BOX_TO_POINT:
                center_x = (boxes[:, 0] + boxes[:, 1]) / 2
                center_y = (boxes[:, 2] + boxes[:, 3]) / 2
                points = torch.stack([center_x, center_y], dim=1)[:, None]
                labels = torch.ones(points.shape[:2], device=samples.device)
                points = (points, labels)

            if config.DISTILL.PROMPT_MASK_TO_POINT:
                point_list = []
                label_list = []
                gt_mask = annos['gt_mask']
                gt_mask = torch.cat(gt_mask, dim=0)
                gt_mask = gt_mask.cuda(non_blocking=True).squeeze(1)
                for g in gt_mask:
                    candidate_indices = g.nonzero()
                    if len(candidate_indices) > 0:
                        selected_index = random.randint(0, len(candidate_indices) - 1)
                        p = candidate_indices[selected_index].flip(0)
                        l = torch.tensor(1, device=samples.device)
                    else:
                        p = torch.zeros(2, device=samples.device)
                        l = torch.tensor(-2, device=samples.device)
                    point_list.append(p)
                    label_list.append(l)
                points = torch.stack(point_list, dim=0)[:, None]
                labels = torch.stack(label_list, dim=0)[:, None]
                points = (points, labels)

            if 'point' not in config.DISTILL.PROMPT_TYPE:
                points = None

            if 'box' not in config.DISTILL.PROMPT_TYPE:
                boxes = None

            cur_prompt_type = config.DISTILL.PROMPT_TYPE
            cur_decoder_iters = config.DISTILL.DECODE_ITERS
            cur_multimask_output = config.DISTILL.MULTIMASK_OUTPUT
            if 'point' in config.DISTILL.PROMPT_TYPE and 'box' in config.DISTILL.PROMPT_TYPE:
                if torch.rand(1) > 0.5:
                    points = None
                    cur_prompt_type = 'box'
                    if not config.DISTILL.ITER_ON_BOX:
                        cur_decoder_iters = 1
                    if not config.DISTILL.MULTIMASK_ON_BOX:
                        cur_multimask_output = 1
                else:
                    boxes = None
                    cur_prompt_type = 'point'

            # Get rid of padding with masking
            valid = torch.zeros(img_bs, cur_multimask_output, *img_size_pad, device=samples.device)
            valid_list = []
            for img_i in range(img_bs):
                h, w = img_size_before_pad[img_i][1:]
                valid[img_i, :, :h, :w] = 1
                valid_list.append(valid[img_i:img_i + 1].expand(num_prompts[img_i], *valid.shape[1:]))
            valid = torch.cat(valid_list, dim=0)

            prev_point = points
            for iter_i in range(cur_decoder_iters):
                if iter_i > 0:
                    with torch.no_grad():
                        valid_down = F.interpolate(valid, mask_s.shape[2:], mode="bilinear", align_corners=False)
                        mask_s = (mask_s.detach() > mask_threshold) * valid_down
                        mask_t = (mask_t.detach() > mask_threshold) * valid_down

                        if mask_t.shape[1] > 1:
                            max_iou_idx = iou_t.argmax(dim=1)
                            batch_range = torch.arange(mask_s.shape[0], device=mask_s.device)
                            mask_s = mask_s[batch_range, max_iou_idx].unsqueeze(1)
                            mask_t = mask_t[batch_range, max_iou_idx].unsqueeze(1)
                        point, label = sample_point_in_mask(mask_s, mask_t, config.DISTILL.POINTS_PER_REFINE_ITER)

                        point[:, :, 0] = point[:, :, 0] / mask_s.shape[3] * img_size_pad[1]
                        point[:, :, 1] = point[:, :, 1] / mask_s.shape[2] * img_size_pad[0]

                        del mask_s, mask_t
                        if prev_point is not None:
                            point = torch.cat([prev_point[0], point], dim=1)
                            label = torch.cat([prev_point[1], label], dim=1)
                        points = (point, label)
                        prev_point = points

                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    sparse_emb_s, dense_emb_s = model(
                        mode='prompt_encoder',
                        points=points, boxes=boxes,
                        masks=masks, num_prompts=num_prompts
                    )
                    mask_s, iou_s, kd_targets_s = model(
                        mode='mask_decoder',
                        image_embeddings=encoder_embeddings,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_emb_s,
                        dense_prompt_embeddings=dense_emb_s,
                        num_multimask_outputs=cur_multimask_output,
                        num_prompts=num_prompts
                    )

                with torch.no_grad():
                    sparse_emb_t, dense_emb_t = teacher_model['prompt_encoder'](
                        points=points, boxes=boxes,
                        masks=masks, num_prompts=num_prompts
                    )
                    mask_t, iou_t, kd_targets_t = teacher_model['mask_decoder'](
                        image_embeddings=saved_embeddings,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_emb_t,
                        dense_prompt_embeddings=dense_emb_t,
                        num_multimask_outputs=cur_multimask_output,
                        num_prompts=num_prompts
                    )

                if VIS:
                    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=samples.device).view(1, 3, 1, 1)
                    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=samples.device).view(1, 3, 1, 1)
                    _samples = (samples * pixel_std + pixel_mean).detach().int()
                    for img_i in range(img_bs):
                        if img_i == 0:
                            cur = slice(0, num_prompts[img_i])
                        else:
                            cur = slice(sum(num_prompts[:img_i]), sum(num_prompts[:img_i + 1]))
                        _boxes = boxes[cur].detach()
                        _mask_t = (mask_t[cur].detach().squeeze(1) > 0).int()
                        _mask_s = (mask_s[cur].detach().squeeze(1) > 0).int()

                        fig = make_fig(_samples[img_i], _boxes, _mask_t, _mask_s, 'gt', 'pred')
                        file_name = annos['info']['file_name']
                        file_name = file_name.split('.')[0]
                        vis_writer.add_figure(f'{file_name}/{iter_i + 1}', fig)

                if config.DISTILL.DECODER_BCE > 0 or config.DISTILL.DECODER_FOCAL > 0 or config.DISTILL.DECODER_DICE > 0:
                    valid_down = F.interpolate(valid, mask_s.shape[2:], mode='bilinear', align_corners=False)
                    _mask_s = mask_s.float()
                    _mask_t = mask_t
                    if config.DISTILL.POINT_REND_SAMPLING:
                        valid_down[valid_down < 0.5] = -torch.inf
                        valid_down[valid_down >= 0.5] = 0
                        with torch.no_grad():
                            point_coords = get_uncertain_point_coords_with_randomness(
                                mask_s + valid_down,
                                lambda logits: calculate_uncertainty(logits),
                                num_points=112 * 112, oversample_ratio=3, importance_sample_ratio=0.75
                            )
                            _mask_t = point_sample(mask_t, point_coords, align_corners=False).squeeze(1)
                        _mask_s = point_sample(mask_s, point_coords, align_corners=False).squeeze(1)
                        valid_down = None

                    temperature = config.DISTILL.TEMPERATURE
                    _mask_s /= temperature
                    _mask_t /= temperature

                    target_logit = True
                    if not config.DISTILL.USE_TEACHER_LOGITS:
                        _mask_t = (_mask_t > mask_threshold).float()
                        target_logit = False

                    if config.DISTILL.DECODER_BCE > 0:
                        _tmp = sigmoid_ce_loss(_mask_s, _mask_t, valid_down,
                                               target_logit) * config.DISTILL.DECODER_BCE / cur_decoder_iters
                        key = f'dec_bce_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                    if config.DISTILL.DECODER_FOCAL > 0:
                        _tmp = sigmoid_focal_loss(_mask_s, _mask_t, valid_down,
                                                  target_logit) * config.DISTILL.DECODER_FOCAL / cur_decoder_iters
                        key = f'dec_focal_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                    if config.DISTILL.DECODER_DICE > 0:
                        _tmp = dice_loss(_mask_s, _mask_t, valid_down,
                                         target_logit) * config.DISTILL.DECODER_DICE / cur_decoder_iters
                        key = f'dec_dice_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                    if config.DISTILL.DECODER_IOU > 0:
                        _tmp = F.mse_loss(iou_s, iou_t) * config.DISTILL.DECODER_IOU / cur_decoder_iters
                        key = f'dec_iou_{iter_i}'
                        loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                if config.DISTILL.DECODER_ATTN > 0:
                    _tmp, count = 0, 0
                    looking_for = ['t2t', 'i2t', 't2i']
                    for key in kd_targets_s:
                        for tgt in looking_for:
                            if tgt in key:
                                count += 1
                                _tmp += F.mse_loss(kd_targets_s[key], kd_targets_t[key]) / cur_decoder_iters
                    _tmp = _tmp / count * config.DISTILL.DECODER_ATTN
                    key = f'dec_attn_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                # The magnitude of the feature and query are very different, so normalization is needed.
                if config.DISTILL.DECODER_FEAT > 0:
                    feat_s = kd_targets_s['feat']
                    feat_t = kd_targets_t['feat']
                    # Cosine similarity (non-linear)
                    # feat_s = F.normalize(feat_s, dim=1)
                    # feat_t = F.normalize(feat_t, dim=1)
                    # _tmp = (1 - torch.einsum('bchw,bchw->bhw', feat_s, feat_t)).mean() * config.DISTILL.DECODER_FEAT

                    # L1 norm (linear)
                    # feat_s = F.normalize(feat_s, dim=1, p=1)
                    # feat_t = F.normalize(feat_s, dim=1, p=1)
                    _tmp = F.mse_loss(feat_s, feat_t) * config.DISTILL.DECODER_FEAT / cur_decoder_iters

                    key = f'dec_feat_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                if config.DISTILL.DECODER_QUERY > 0:
                    query_s = kd_targets_s['query']
                    query_t = kd_targets_t['query']
                    # Cosine similarity (non-linear)
                    # query_s = F.normalize(query_s, dim=-1)
                    # query_t = F.normalize(query_t, dim=-1)
                    # _tmp = (1 - torch.einsum('bnc,bnc->bn', query_s, query_t)).mean() * config.DISTILL.DECODER_QUERY

                    # L1 norm (linear)
                    query_s = F.normalize(query_s, dim=1, p=1)
                    query_t = F.normalize(query_t, dim=1, p=1)
                    _tmp = F.mse_loss(query_s, query_t) * config.DISTILL.DECODER_QUERY / cur_decoder_iters

                    key = f'dec_query_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

        for key in loss:
            loss[key] = loss[key] / config.TRAIN.ACCUMULATION_STEPS
            meters[key].update(loss[key].item(), len(samples))

        total_loss = sum(loss.values())

        if loss_writer is not None:
            display_dict = {'total': total_loss}
            for key in loss:
                display_dict[key] = loss[key].item()

            loss_writer.add_scalars('loss', display_dict, epoch * num_steps + idx)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        torch.cuda.synchronize()

        loss_meter.update(total_loss.item(), len(samples))
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    extra_meters_str = f'Train-Summary: [{epoch}/{config.TRAIN.EPOCHS}]\t'
    for k, v in meters.items():
        v.sync()
        extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
    logger.info(extra_meters_str)
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()
    config.defrost()
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        ddp_backend = 'gloo'
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
        ddp_backend = 'nccl'

    torch.distributed.init_process_group(
        backend=ddp_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
                       config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
                              config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
                           config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        config_dict = dict(config)
        config_dict['git'] = get_git_info()
        if args.use_wandb:
            wandb_output_path = config.OUTPUT
            wandb.init(project="EdgeSAM", config=config_dict,
                       dir=wandb_output_path)

    # print git info
    logger.info('===== git =====')
    logger.info(str(get_git_info()))

    # print config
    logger.info(config.dump())

    main(args, config)
