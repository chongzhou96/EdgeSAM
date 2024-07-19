# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from typing import Any, List

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .sam import Sam


class SamBatch(Sam):

    def __init__(
            self,
            image_encoder,
            prompt_encoder,
            mask_decoder,
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
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std, rpn_head)

    def forward(self, mode, **kwargs):
        if mode == 'image_encoder':
            return self.image_encoder(**kwargs)
        elif mode == 'prompt_encoder':
            return self.prompt_encoder(**kwargs)
        elif mode == 'mask_decoder':
            return self.mask_decoder(**kwargs)
        elif mode == 'full':
            encoder_embeddings = self.image_encoder(kwargs['x'])
            sparse_emb_s, dense_emb_s = self.prompt_encoder(kwargs['points'], kwargs['boxes'],
                                                            kwargs['masks'], kwargs['num_prompts'])
            dense_pe = self.prompt_encoder.get_dense_pe()
            return self.mask_decoder(encoder_embeddings, dense_pe, sparse_emb_s, dense_emb_s,
                                     kwargs['multimask_output'], kwargs['num_prompts'])


class PromptEncoderBatch(PromptEncoder):

    def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation=nn.GELU):
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)

    def forward(self, points, boxes, masks, num_prompts, box_labels=None):
        total_prompts = sum(num_prompts)
        sparse_embeddings = torch.empty((total_prompts, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            if box_labels is not None:
                box_embeddings[box_labels == -1, :, :] = 0.0
                box_embeddings[box_labels == -1, :, :] += self.not_a_point_embed.weight
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                total_prompts, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class MaskDecoderBatch(MaskDecoder):

    def predict_masks(
            self,
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            num_prompts=None,
            kd_targets=None
    ):
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        img_emd_list = []
        for i, num in enumerate(num_prompts):
            img_emd = image_embeddings[i:i + 1].expand(num, *image_embeddings.size()[1:])
            img_emd_list.append(img_emd)

        src = torch.cat(img_emd_list, dim=0)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(sum(num_prompts), *image_pe.size()[1:])

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, kd_targets)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.yield_kd_targets:
            kd_targets['query'] = hyper_in
            kd_targets['feat'] = upscaled_embedding
        return masks, iou_pred
