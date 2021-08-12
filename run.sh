kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --max_aug_per_video=5 --multiprocessing-distributed --epochs=30 --checkpoint_dir='/disk/scratch_fast/s2004019/youcook2/checkpoints/procnet/run7_iou/'



