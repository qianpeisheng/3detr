CUDA_VISIBLE_DEVICES=1 python3 main.py \
--dataset_name sunrgbd \
--nqueries 128 \
--test_ckpt /home/peisheng/3detr/ckpts_sunrgbd_pretrained/sunrgbd_masked_ep1080.pth \
--test_only \
--enc_type masked