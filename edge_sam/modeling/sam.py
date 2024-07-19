# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
from mmdet.models.necks import FPN
from projects.EfficientDet import efficientdet
from mmengine import ConfigDict

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        rpn_head=None
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.rpn_head = None
        self.fpn = None
        if rpn_head == 'centernet':
            self.fpn = FPN(
                in_channels=[96, 192, 384],
                out_channels=96,
                num_outs=5
            )
            self.rpn_head = CenterNetUpdateHead(
                num_classes=1,
                in_channels=96,
                stacked_convs=4,
                feat_channels=96,
                strides=[8, 16, 32, 64, 128]
            )
        elif rpn_head == 'rpn':
            self.fpn = FPN(
                in_channels=[96, 192, 384],
                out_channels=96,
                num_outs=5
            )
            self.rpn_head = RPNHead(
                in_channels=96,
                feat_channels=96,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
            )
        elif rpn_head == 'efficient_det':
            norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)
            self.fpn = efficientdet.BiFPN(
                num_stages=3,
                in_channels=[96, 192, 384],
                out_channels=64,
                start_level=0,
                norm_cfg=norm_cfg
            )
            self.rpn_head = efficientdet.EfficientDetSepBNHead(
                num_classes=1,
                num_ins=5,
                in_channels=64,
                feat_channels=64,
                stacked_convs=3,
                norm_cfg=norm_cfg,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[1.0, 0.5, 2.0],
                    strides=[8, 16, 32, 64, 128],
                    center_offset=0.5),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0])
            )
        self.use_rpn = self.rpn_head is not None

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # For FLOPs, params count, and speed test
    @torch.no_grad()
    def forward_dummy_encoder(self, x):
        image_encoder_outs = self.image_encoder(x)
        outs = (image_encoder_outs,)
        if self.use_rpn:
            image_embeddings = image_encoder_outs[-1]
            proposals = self.forward_rpn(image_encoder_outs[:-1])
            outs += (proposals[0].bboxes, proposals[0].scores)
        else:
            image_embeddings = image_encoder_outs
        return outs

    # For FLOPs and params count
    @torch.no_grad()
    def forward_dummy_decoder(self, image_embeddings, points):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=None, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=1,
        )
        return low_res_masks, iou_predictions

    @torch.no_grad()
    def forward_rpn(self, features, score_thr=0.05, with_nms=False):
        fpn_out = self.fpn(features)
        rpn_out = self.rpn_head(fpn_out)

        batch_size = features[0].shape[0]
        batch_img_metas = [dict(
            img_shape=(1024, 1024)
        )] * batch_size

        cfg = ConfigDict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=score_thr,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1000)

        predictions = self.rpn_head.predict_by_feat(
            *rpn_out, batch_img_metas=batch_img_metas, with_nms=with_nms, rescale=False, cfg=cfg)
        return predictions

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        num_multimask_outputs: int = 1,
        use_stability_score: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks. Choices: 1, 3, 4.
          use_stability_score (bool): If true, use stability scores to substitute
            IoU predictions.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_encoder_outs = self.image_encoder(input_images)
        if self.use_rpn:
            image_embeddings = image_encoder_outs[-1]
            proposals = self.forward_rpn(image_encoder_outs[:-1])
        else:
            image_embeddings = image_encoder_outs

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                num_multimask_outputs=num_multimask_outputs,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
