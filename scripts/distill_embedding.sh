#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
#   training/train.py --cfg training/configs/rep_vit_m1_fuse_sa_distill.yaml \
#   --output ./output/ \
#   --batch-size 2

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
#   training/train.py --cfg training/configs/rep_vit_m1_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml \
#   --output ./output/ \
#   --batch-size 2

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 \
#   training/train.py --cfg training/configs/rep_vit_m1_fuse_sa_distill.yaml \
#   --output ./output/ \
#   --batch-size 2 \
#   --opt DATA.DEBUG True TRAIN.EPOCHS 1 DISTILL.MAX_ALLOWED_PROMPTS 8

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 \
   training/train.py --cfg training/configs/rep_vit_m1_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml \
   --output ./output/ \
   --batch-size 2 \
   --opt DATA.DEBUG True TRAIN.EPOCHS 1 DISTILL.MAX_ALLOWED_PROMPTS 8