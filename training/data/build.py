# --------------------------------------------------------
# EdgeSAM Data Builder
# Based on the code: TinyViT
#   (https://github.com/microsoft/Cream/tree/main/TinyViT)
# --------------------------------------------------------

import torch
import torch.distributed as dist
from mmengine.dataset import pseudo_collate

from .augmentation.dataset_wrapper import DatasetWrapper
from .sa1b_dataset import SA1BDataset
from .coco_dataset import COCODataset
from .sampler import MyDistributedSampler


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config)
    config.freeze()

    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    sampler_train = MyDistributedSampler(
        dataset_train, shuffle=True,
        drop_last=False, padding=True, pair=False,
    )

    sampler_val = MyDistributedSampler(
        dataset_val, shuffle=False,
        drop_last=False, padding=False, pair=False,
    )

    # EdgeSAM Dataset Wrapper
    dataset_train = DatasetWrapper(dataset_train,
                                   logits_path=config.DISTILL.TEACHER_EMBED_PATH,
                                   topk=config.DISTILL.EMBED_DIM,
                                   write=config.DISTILL.SAVE_TEACHER_EMBED,
                                   num_embedding=64*64,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # modified for EdgeSAM, we save image embeddings of all samples
        drop_last=not config.DISTILL.SAVE_TEACHER_EMBED,
        collate_fn=pseudo_collate
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        collate_fn=pseudo_collate
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_dataset(is_train, config):
    if config.DATA.DATASET == 'sa1b':
        num_samples = 100 if config.DATA.DEBUG else config.DATA.NUM_SAMPLES
        dataset = SA1BDataset(data_root=config.DATA.DATA_PATH, split='train' if is_train else 'val',
                               num_samples=num_samples, filter_by_area=config.DATA.FILTER_BY_AREA,
                               max_allowed_prompts=config.DISTILL.MAX_ALLOWED_PROMPTS, fix_seed=False,
                               load_gt_mask=config.DATA.LOAD_GT_MASK,
                               mask_nms_thresh=config.DATA.MASK_NMS,
                               box_jitter=config.DATA.BOX_JITTER)
        nb_classes = 0
    elif config.DATA.DATASET in ['coco', 'cocofied_lvis', 'lvis']:
        num_samples = 100 if config.DATA.DEBUG else -1
        dataset = COCODataset(data_root=config.DATA.DATA_PATH, split='train' if is_train else 'val',
                              num_samples=num_samples, filter_by_area=config.DATA.FILTER_BY_AREA,
                              max_allowed_prompts=config.DISTILL.MAX_ALLOWED_PROMPTS, fix_seed=False,
                              load_gt_mask=config.DATA.LOAD_GT_MASK, annotation=config.DATA.DATASET)
        nb_classes = 0
    else:
        raise NotImplementedError("We only support ImageNet, SA-1B, and COCO Now.")

    return dataset, nb_classes