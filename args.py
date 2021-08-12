import argparse


def get_args(description='Youtube-Text-Video'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--feature_root',
        type=str,
        default='/disk/scratch_fast/s2004019/youcook2/raw_videos',
        help='train csv')
    parser.add_argument(
        '--yc2_dur_file',
        type=str,
        default='/disk/scratch_fast/s2004019/youcook2/yc2/yc2_duration_frame.csv',
        help='train csv')
    parser.add_argument(
        '--yc2_annotation_file',
        type=str,
        default='/disk/scratch_fast/s2004019/youcook2/yc2/yc2_new_annotations_trainval_test.json',
        help='train csv')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='/disk/scratch_fast/s2004019/youcook2/checkpoints/procnet/run4/',
        help='checkpoint model folder')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-file', default='dist-file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--model_name_postfix', default='_iou', type=str,
                        help='model psot fix after `model_best_<>.pth.tar`')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--eval_test', dest='eval_test', action='store_true',
                        help='Evaluate Test Set Only')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='number of frames to be sampled from a video')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='number of frames to be sampled from a video')
    parser.add_argument('--max_aug_per_video', type=int, default=10,
                        help='number of frames to be sampled from a video')
    parser.add_argument('--frames_per_video', type=int, default=500,
                        help='number of frames to be sampled from a video')
    parser.add_argument('--classes', type=int, default=2,
                        help='embedding dim')
    parser.add_argument('--video_dim', type=int, default=2048,
                        help='embedding dim')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='BiLSTM hidden_size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=1,
                        help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=10,
                        help='Information display frequence')
    parser.add_argument('--seed', type=int, default=8,
                        help='random seed')
    parser.add_argument('--verbose', type=int, default=1,
                        help='')
    args = parser.parse_args()
    return args
