import os.path
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from PIL import Image
import json
from pycocotools import mask as mask_utils
from edge_sam.utils.transforms import ResizeLongestSide
from edge_sam.utils.common import xywh2xyxy
import copy


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, data_root, img_size=1024, split='train', num_samples=-1,
                 sort_by_area=False, filter_by_area=None,
                 pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                 load_gt_mask=False, max_allowed_prompts=-1, fix_seed=False,
                 annotation='coco'):
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
        self.annotation = annotation

        self.num_samples = num_samples
        self.split = split
        self.prepare_data()

    def prepare_data(self):
        self.data = []
        self.keys = []

        if self.annotation == 'coco':
            anno_path = f'{self.data_root}/annotations/instances_{self.split}2017.json'
        elif self.annotation == 'cocofied_lvis':
            anno_path = f'{self.data_root}/annotations/lvis_v1_{self.split}_cocofied.json'
        elif self.annotation == 'lvis':
            anno_path = f'{self.data_root}/annotations/lvis_v1_{self.split}.json'
        with open(anno_path, 'r') as f:
            anno_json = json.load(f)

        ignore_path = f'{self.data_root}/annotations/ignore.json'
        if os.path.exists(ignore_path):
            with open(ignore_path, 'r') as f:
                self.ignore_list = json.load(f)
        else:
            self.ignore_list = []

        imgs = dict()
        for img_info in anno_json['images']:
            img_id = img_info['id']
            imgs[img_id] = img_info

        annos = dict()
        for anno_info in anno_json['annotations']:
            if 'iscrowd' in anno_info and anno_info['iscrowd']:
                continue
            img_id = anno_info['image_id']
            if img_id not in annos:
                annos[img_id] = [anno_info]
            else:
                annos[img_id].append(anno_info)

        counter = 0
        for img_id in imgs:
            if img_id in annos:
                if 'file_name' in imgs[img_id]:
                    file_name = imgs[img_id]['file_name']
                else:
                    file_name = imgs[img_id]['coco_url'].split('/')[-1]

                if file_name in self.ignore_list:
                    continue

                self.keys.append(file_name)
                self.data.append((imgs[img_id], annos[img_id]))

                counter += 1
                if self.num_samples > 0:
                    if counter == self.num_samples:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_info, anno_info = copy.deepcopy(self.data[idx])
        file_name = self.keys[idx]
        img_path = f"{self.data_root}/trainval/{file_name}"
        img = Image.open(img_path).convert('RGB')

        img = pil_to_tensor(img)
        original_size = img.shape[1:]

        prompt_box_list = []
        if self.load_gt_mask:
            gt_mask_list = []
        mask_area = []

        for record in anno_info:
            prompt_box = np.asarray(record['bbox'])
            prompt_box_list.append(prompt_box)

            if self.load_gt_mask:
                h, w = img_info['height'], img_info['width']
                segm = record['segmentation']
                if type(segm) == list:
                    # polygon -- a single object might consist of multiple parts
                    # we merge all parts into one mask rle code
                    rles = mask_utils.frPyObjects(segm, h, w)
                    rle = mask_utils.merge(rles)
                elif type(segm['counts']) == list:
                    # uncompressed RLE
                    rle = mask_utils.frPyObjects(segm, h, w)
                else:
                    # rle
                    rle = segm
                gt_mask = mask_utils.decode(rle)
                gt_mask_list.append(gt_mask)

            mask_area.append(record['area'])

        mask_area = np.array(mask_area)

        prompt_box = np.stack(prompt_box_list, axis=0)
        prompt_box = torch.from_numpy(prompt_box)

        if self.load_gt_mask:
            gt_mask = np.stack(gt_mask_list, axis=0)
            gt_mask = torch.from_numpy(gt_mask)

        if self.sort_by_area:
            area = prompt_box[:, 2] * prompt_box[:, 3]
            indices = torch.argsort(area, descending=True)
            prompt_box = prompt_box.gather(dim=0, index=indices[:, None].expand(prompt_box.shape))
            if self.load_gt_mask:
                gt_mask = gt_mask.gather(dim=0, index=indices[:, None, None].expand(gt_mask.shape))

        img = self.transform.apply_image_torch(img[None].float()).squeeze(0)
        prompt_box = self.transform.apply_boxes_torch(prompt_box, original_size)
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
                if self.load_gt_mask:
                    gt_mask = gt_mask[selected]

        num_prompts = prompt_box.shape[0]
        if num_prompts > self.max_allowed_prompts > 0:
            # for evaluation reproducibility
            if self.fix_seed:
                torch.manual_seed(idx)
            selected = torch.randint(0, num_prompts, (self.max_allowed_prompts,))
            prompt_box = prompt_box[selected]
            if self.load_gt_mask:
                gt_mask = gt_mask[selected]

        prompt_box = xywh2xyxy(prompt_box)

        img_info_new = dict(file_name=file_name)
        for key in img_info:
            if key in ['height', 'width']:
                img_info_new[key] = img_info[key]

        anno = dict(prompt_box=prompt_box, info=img_info_new, img_size_before_pad=img_size_before_pad)
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
    data_root = '../../data/coco/'
    # dataset = COCODataset(data_root, num_samples=100, filter_by_area=[None, None], sort_by_area=True, load_gt_mask=True)
    # num_img = len(dataset)
    # print(f'number of images: {num_img}')

    # gt_mask_area = torch.empty(0)
    # for img, anno in dataset:
    #     gt_mask = anno['gt_mask']
    #     area = gt_mask.flatten(1).sum(1)
    #     gt_mask_area = torch.cat([gt_mask_area, area])

    # num_gt_mask = int(gt_mask_area.shape[0])
    # print(f'number of masks: {num_gt_mask}')
    # print(f'masks per image: {num_gt_mask/num_img}')
    # print(f'area mean: {gt_mask_area.mean():.2f}')
    # print(f'area std: {gt_mask_area.std():.2f}')

    # gt_box_area = torch.empty(0)
    # for img, anno in dataset:
    #     gt_box = anno['prompt_box']
    #     area = (gt_box[:, 2]- gt_box[:, 0]) * (gt_box[:, 3]- gt_box[:, 1])
    #     gt_box_area = torch.cat([gt_box_area, area])
    #
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
    #     break
    # print(dataset.get_keys())

    # coco = COCODataset(data_root)
    # coco_keys = coco.get_keys()
    # lvis = COCODataset(data_root, annotation='lvis')
    # lvis_keys = lvis.get_keys()

    # print('not in lvis:', len(set(coco_keys)-set(lvis_keys)))
    # print('not in coco:', len(set(lvis_keys)-set(coco_keys)))
    # with open(f'{data_root}/annotations/ignore.json', 'w') as f:
    #     json.dump(list(set(lvis_keys)-set(coco_keys)), f)