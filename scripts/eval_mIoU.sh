all_args=("$@")
rest_args=("${all_args[@]:2}")


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples 1000 \
    --max-prompt-bs 64 \
    --img-bs 1 \
    --refine-iter 3 \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples 1000 \
    --max-prompt-bs 64 \
    --img-bs 1 \
    --prompt-types 'point' \
    --point-from 'mask-center' \
    --refine-iter 3 \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --img-bs 1 \
    --dataset 'coco' \
    --refine-iter 3 \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --dataset 'coco' \
    --point-from 'mask-center' \
    --refine-iter 3 \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --refine-iter 1 \
    --img-bs 1 \
    --dataset 'lvis' \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --refine-iter 1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --dataset 'lvis' \
    --point-from 'mask-center' \
    ${rest_args[@]}

# ---------------------------------------------

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples 1000 \
    --max-prompt-bs 64 \
    --refine-iter 1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --point-from 'mask-center' \
    --num-multimask-outputs 3 \
    --multimask-select 'score' \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples 1000 \
    --max-prompt-bs 64 \
    --refine-iter 1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --point-from 'mask-center' \
    --num-multimask-outputs 3 \
    --multimask-select 'oracle' \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --refine-iter 1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --dataset 'coco' \
    --point-from 'mask-center' \
    --num-multimask-outputs 3 \
    --multimask-select 'score' \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --refine-iter 1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --dataset 'coco' \
    --point-from 'mask-center' \
    --num-multimask-outputs 3 \
    --multimask-select 'oracle' \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --refine-iter 1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --dataset 'lvis' \
    --point-from 'mask-center' \
    --num-multimask-outputs 3 \
    --multimask-select 'score' \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --refine-iter 1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --dataset 'lvis' \
    --point-from 'mask-center' \
    --num-multimask-outputs 3 \
    --multimask-select 'oracle' \
    ${rest_args[@]}
