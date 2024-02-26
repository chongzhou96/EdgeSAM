# --------------------------------------------------------
# Save Teacher Embeddings
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

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from edge_sam import get_config, build_sam_from_config

from my_meter import AverageMeter
from data import build_loader
from logger import create_logger
from utils import load_checkpoint, NativeScalerWithGradNormCount, add_common_args


def parse_option():
    parser = argparse.ArgumentParser(
        'EdgeSAM saving teacher image embedding script', add_help=False)
    add_common_args(parser)
    parser.add_argument('--check-saved-embed',
                        action='store_true', help='Check saved embeddings')
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_sam_from_config(config)
    model.cuda()

    if not os.path.exists(config.DISTILL.TEACHER_EMBED_PATH):
        os.makedirs(config.DISTILL.TEACHER_EMBED_PATH)

    logger.info(str(model))

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    optimizer = None
    lr_scheduler = None

    assert config.MODEL.RESUME
    loss_scaler = NativeScalerWithGradNormCount()
    load_checkpoint(config, model_without_ddp, optimizer,
                    lr_scheduler, loss_scaler, logger)

    if args.check_saved_embed:
        logger.info("Start checking embeddings")
    else:
        logger.info("Start saving embeddings")

    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        dataset_train.set_epoch(epoch)
        data_loader_train.sampler.set_epoch(epoch)

        if args.check_saved_embed:
            check_embeddings_one_epoch(config, model, data_loader_train, epoch)
        else:
            save_embeddings_one_epoch(config, model, data_loader_train, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Saving embeddings time {}'.format(total_time_str))


@torch.no_grad()
def save_embeddings_one_epoch(config, model, data_loader, epoch):
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    file_manager = data_loader.dataset.get_manager()

    for idx, ((samples, _), (keys, seeds)) in enumerate(data_loader):
        samples = torch.stack(samples, dim=0).cuda(non_blocking=True)
        samples = samples.cuda(non_blocking=True)
        seeds = np.stack(seeds, axis=0)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        torch.cuda.synchronize()

        write_tic = time.time()

        cpu_device = torch.device('cpu')
        outputs = outputs.detach().to(device=cpu_device, dtype=torch.float16)

        # seeds = seeds.numpy()
        outputs = outputs.numpy()

        # check data type
        assert seeds.dtype == np.int32, seeds.dtype
        assert outputs.dtype == np.float16, outputs.dtype

        for key, seed, output in zip(keys, seeds, outputs):
            bstr = seed.tobytes() + output.tobytes()
            file_manager.write(key, bstr)
        meters['write_time'].update(time.time() - write_tic)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Save: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} save image embeddings takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def check_embeddings_one_epoch(config, model, data_loader, epoch):
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    for idx, ((samples, _), (saved_embeddings, seeds)) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        if saved_embeddings.size() != outputs.size():
            saved_embeddings = saved_embeddings.reshape(outputs.shape)

        torch.cuda.synchronize()

        meters['error'].update(
            (outputs - saved_embeddings.cuda()).abs().mean().item())

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            extra_meters_str = ''
            for k, v in meters.items():
                extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})\t'
            logger.info(
                f'Check: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'{extra_meters_str}'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} check image embeddings takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()

    config.defrost()
    assert len(
        config.DISTILL.TEACHER_EMBED_PATH) > 0, "Please fill in the config DISTILL.TEACHER_EMBED_PATH"
    config.DISTILL.ENABLED = True
    if not args.check_saved_embed:
        config.DISTILL.SAVE_TEACHER_EMBED = True
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # The seed changes with config, rank, world_size and epoch
    seed = config.SEED + dist.get_rank() + config.TRAIN.START_EPOCH * \
        dist.get_world_size()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
