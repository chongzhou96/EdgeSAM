import os.path

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import glob
from pathlib import Path
from PIL import Image
import json
from pycocotools import mask as mask_utils
from edge_sam.utils.transforms import ResizeLongestSide
from edge_sam.utils.common import xywh2xyxy, xyxy2xywh
import copy
from mmengine.dataset import pseudo_collate
from tqdm import tqdm
from PIL import ImageFile


class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, img_size=1024, split='train', num_samples=-1,
                 sort_by_area=False, filter_by_area=None,
                 pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                 load_gt_mask=False, max_allowed_prompts=-1, fix_seed=False, mask_nms_thresh=-1.,
                 box_jitter=False):
        super().__init__()
        self.data_root = data_root
        self.img_size = img_size
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.transform = ResizeLongestSide(img_size)
        self.sort_by_area = sort_by_area
        self.filter_by_area = filter_by_area
        self.load_gt_mask = load_gt_mask
        self.max_allowed_prompts = max_allowed_prompts
        self.fix_seed = fix_seed
        self.mask_nms_thresh = mask_nms_thresh
        self.box_jitter = box_jitter

        self.num_samples = num_samples
        self.split = split
        self.prepare_data()
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def prepare_data(self):
        self.data = []
        self.keys = []

        counter = 0
        for img_path in list(glob.glob(f'{self.data_root}/images/{self.split}/*.jpg')):
            name = Path(img_path).stem
            anno_path = f'{self.data_root}/annotations/{self.split}/{name}.json'
            if not os.path.exists(anno_path):
                continue

            self.data.append((img_path, anno_path))
            self.keys.append(name)

            counter += 1
            if self.num_samples > 0:
                if counter == self.num_samples:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, anno_path = copy.deepcopy(self.data[idx])
        img = Image.open(img_path).convert('RGB')
        img = pil_to_tensor(img)
        original_size = img.shape[1:]

        with open(anno_path, 'r') as f:
            anno_json = json.load(f)
            anno_raw = anno_json['annotations']

        height, width = anno_json['image']['height'], anno_json['image']['width']
        prompt_box_list = []
        prompt_point_list = []
        mask_area = []

        if self.load_gt_mask:
            gt_mask_list = []
        for record in anno_raw:
            prompt_box = np.asarray(record['bbox'])
            prompt_box_list.append(prompt_box)

            prompt_point = np.asarray(record['point_coords'])
            prompt_point_list.append(prompt_point)

            if self.load_gt_mask:
                segm = record['segmentation']
                if type(segm) == list:
                    # polygon -- a single object might consist of multiple parts
                    # we merge all parts into one mask rle code
                    rles = mask_utils.frPyObjects(segm, height, width)
                    rle = mask_utils.merge(rles)
                elif type(segm['counts']) == list:
                    # uncompressed RLE
                    rle = mask_utils.frPyObjects(segm, height, width)
                else:
                    # rle
                    rle = segm
                gt_mask = mask_utils.decode(rle)
                gt_mask_list.append(gt_mask)

            mask_area.append(record['area'])

        mask_area = np.array(mask_area)

        if self.mask_nms_thresh > 0 and self.load_gt_mask:
            tot_mask = np.zeros_like(gt_mask_list[0])
            ignore_flags = []
            sort_idx = np.argsort(-mask_area)

            for i in sort_idx:
                gt_mask = gt_mask_list[i]
                ratio = (gt_mask * tot_mask).sum() / gt_mask.sum()
                if ratio > self.mask_nms_thresh:
                    ignore_flags.append(True)
                else:
                    ignore_flags.append(False)
                    tot_mask = (tot_mask + gt_mask).clip(max=1)

            b_list, p_list, m_list = [], [], []
            for i, ignore_flag in zip(sort_idx, ignore_flags):
                if not ignore_flag:
                    b_list.append(prompt_box_list[i])
                    p_list.append(prompt_point_list[i])
                    m_list.append(gt_mask_list[i])
            prompt_box_list = b_list
            prompt_point_list = p_list
            gt_mask_list = m_list

        prompt_box = np.stack(prompt_box_list, axis=0)
        prompt_box = torch.from_numpy(prompt_box)

        prompt_point = np.stack(prompt_point_list, axis=0)
        prompt_point = torch.from_numpy(prompt_point)

        if self.load_gt_mask:
            gt_mask = np.stack(gt_mask_list, axis=0)
            gt_mask = torch.from_numpy(gt_mask)

        if self.box_jitter:
            prompt_box = xywh2xyxy(prompt_box)
            N = prompt_box.shape[0]
            delta_w = (prompt_box[:, 2] - prompt_box[:, 0])[:, None] * 0.1 * torch.randn(N, 2)
            delta_h = (prompt_box[:, 3] - prompt_box[:, 1])[:, None] * 0.1 * torch.randn(N, 2)
            delta_w = delta_w.clip(min=-20, max=20)
            delta_h = delta_h.clip(min=-20, max=20)
            prompt_box[:, 0::2] = (prompt_box[:, 0::2] + delta_w).clip(min=0, max=width)
            prompt_box[:, 1::2] = (prompt_box[:, 1::2] + delta_h).clip(min=0, max=height)
            prompt_box = xyxy2xywh(prompt_box)

        if self.sort_by_area:
            area = prompt_box[:, 2] * prompt_box[:, 3]
            indices = torch.argsort(area, descending=True)
            prompt_box = prompt_box.gather(dim=0, index=indices[:, None].expand(prompt_box.shape))
            prompt_point = prompt_point.gather(dim=0, index=indices[:, None, None].expand(prompt_point.shape))
            if self.load_gt_mask:
                gt_mask = gt_mask.gather(dim=0, index=indices[:, None, None].expand(gt_mask.shape))

        img = self.transform.apply_image_torch(img[None].float()).squeeze(0)
        prompt_box = self.transform.apply_boxes_torch(prompt_box, original_size)
        prompt_point = self.transform.apply_coords_torch(prompt_point, original_size)
        if self.load_gt_mask:
            gt_mask = self.transform.apply_masks_torch(gt_mask, original_size)

        # make it Tensor to avoid collate_fn's stacking
        img_size_before_pad = torch.tensor(img.shape, device=img.device)
        img = self.pad(self.norm(img))
        if self.load_gt_mask:
            gt_mask = self.pad(gt_mask)

        if self.filter_by_area is not None:
            area_min = eval(str(self.filter_by_area[0]))
            area_max = eval(str(self.filter_by_area[1]))
            if area_min is None:
                area_min = -torch.inf
            if area_max is None:
                area_max = torch.inf

            area = prompt_box[:, 2] * prompt_box[:, 3]
            selected = (area >= area_min) & (area <= area_max)
            if selected.sum() > 0:
                prompt_box = prompt_box[selected]
                prompt_point = prompt_point[selected]
                if self.load_gt_mask:
                    gt_mask = gt_mask[selected]

        num_prompts = prompt_box.shape[0]
        if num_prompts > self.max_allowed_prompts > 0:
            # for evaluation reproducibility
            if self.fix_seed:
                torch.manual_seed(idx)
            selected = torch.randint(0, num_prompts, (self.max_allowed_prompts,))
            prompt_box = prompt_box[selected]
            prompt_point = prompt_point[selected]
            if self.load_gt_mask:
                gt_mask = gt_mask[selected]

        prompt_box = xywh2xyxy(prompt_box)

        anno = dict(
            prompt_box=prompt_box, prompt_point=prompt_point,
            info=anno_json['image'], img_size_before_pad=img_size_before_pad
        )
        if self.load_gt_mask:
            anno['gt_mask'] = gt_mask

        return img, anno

    def get_keys(self):
        return self.keys

    def norm(self, x):
        """Normalize pixel values"""
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def pad(self, x):
        """Pad to a square input"""
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


if __name__ == '__main__':
    data_root = '../../data/SAM/'
    dataset = SA1BDataset(data_root, num_samples=100, filter_by_area=[64 * 64, None], sort_by_area=False,
                           load_gt_mask=True, mask_nms_thresh=0.8, box_jitter=True)
    num_img = len(dataset)
    print(f'number of images: {num_img}')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=8, collate_fn=pseudo_collate)

    gt_mask_area = torch.empty(0)
    for img, anno in tqdm(dataloader):
        gt_mask = anno['gt_mask'][0]
        area = gt_mask.flatten(1).sum(1)
        gt_mask_area = torch.cat([gt_mask_area, area])

    num_gt_mask = int(gt_mask_area.shape[0])
    print(f'number of masks: {num_gt_mask}')
    print(f'masks per image: {num_gt_mask / num_img}')
    print(f'area mean: {gt_mask_area.mean():.2f}')
    print(f'area std: {gt_mask_area.std():.2f}')

    # gt_box_area = torch.empty(0)
    # for img, anno in dataset:
    #     gt_box = anno['prompt_box']
    #     area = (gt_box[:, 2]-gt_box[:, 0]) * (gt_box[:, 3]-gt_box[:, 1])
    #     gt_box_area = torch.cat([gt_box_area, area])

    # num_gt_box = int(gt_box_area.shape[0])
    # print(f'number of boxes: {num_gt_box}')
    # print(f'boxes per image: {num_gt_box/num_img}')
    # print(f'area mean: {gt_box_area.mean():.2f}')
    # print(f'area std: {gt_box_area.std():.2f}')
    # print(torch.histogram(gt_box_area, torch.tensor([8*8, 16*16, 32*32, 64*64, 128*128, 256*256]).float()))

    # for img, anno in dataset:
    #     print(img.shape)
    #     print(anno['gt_mask'].shape)
    #     print(anno['prompt_box'].shape)
    #     print(anno['prompt_point'].shape)
    #     break
    # print(dataset.get_keys())