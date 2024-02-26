CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
    training/save_embedding.py --cfg training/configs/teacher/sam_vit_huge_sa1b.yaml \
    --batch-size 8 \
    --eval \
    --resume weights/sam_vit_h_4b8939.pth

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 \
#    training/save_embedding.py --cfg training/configs/teacher/sam_vit_huge_sa1b.yaml \
#    --batch-size 1 \
#    --eval \
#    --resume weights/sam_vit_h_4b8939.pth \
#    --opt DATA.DEBUG True