from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import random

import numpy as np
import torch as th
from torch._C import device, dtype
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

from args import get_args
from dataset import YC2ProcNetDataset
from loss import ProcNetLoss
from models.model import ProcNet
from utils import AllGather

allgather = AllGather.apply


def main(args):

    args.log_file_path = os.path.join(
        args.checkpoint_dir, "logs{}.txt".format(args.model_name_postfix))
    mode = "w"
    if args.resume:
        mode = "a"
    if args.local_rank == 0:
        with open(args.log_file_path, mode) as f:
            f.write("=="*100)
            f.write("\nInitiaing The training process.\n")

    args.distributed = args.world_size == -1 or args.multiprocessing_distributed
    args.gpu = args.local_rank if args.local_rank > -1 else None
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
        )
        args.world_size = torch.distributed.get_world_size()

    model = ProcNet(input_encoding_size=512, rnn_size=512, clip_number=16,
                    kernelInfo=[3, 123, 8], frames_per_video=args.frames_per_video,
                    mp_scale=(8, 5), video_feat=512)
    if args.eval_test:
        test_dataset = YC2ProcNetDataset(feature_root=args.feature_root,
                                         data_file="validation_frames.json",
                                         dur_file=args.yc2_dur_file,
                                         annotation_file=args.yc2_annotation_file,
                                         split="testing", frames_per_video=args.frames_per_video,
                                         max_augs=args.max_aug_per_video,)
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=8)
    else:
        train_dataset = YC2ProcNetDataset(feature_root=args.feature_root,
                                          data_file="training_frames.json",
                                          dur_file=args.yc2_dur_file,
                                          annotation_file=args.yc2_annotation_file,
                                          split="training", frames_per_video=args.frames_per_video,
                                          max_augs=args.max_aug_per_video, max_samples=args.max_samples)

        valid_dataset = YC2ProcNetDataset(feature_root=args.feature_root,
                                          data_file="validation_frames.json",
                                          dur_file=args.yc2_dur_file,
                                          annotation_file=args.yc2_annotation_file,
                                          split="validation", frames_per_video=args.frames_per_video)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.distributed and not args.eval_test:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset)
    else:
        train_sampler = None
        test_sampler = None
    if not args.eval_test:
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=(train_sampler is None), num_workers=8, sampler=train_sampler)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=1, shuffle=False, num_workers=8, sampler=test_sampler)

        # Optimizers + Loss
        criterion = ProcNetLoss(samples=200, kernelInfo=[3, 123, 8])
        criterion.cuda(args.gpu)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(
        0.8, 0.999), eps=1e-08, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # optionally resume from a checkpoint
    checkpoint_path = os.path.join(
        args.checkpoint_dir, "model_best_iou.pth.tar")
    best_iou = 0.
    if args.resume:
        if checkpoint_path:
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_iou = checkpoint["miou"]
            log("=> loaded checkpoint '{}' (epoch {}) with best mIoU : {}".format(
                checkpoint_path, checkpoint["epoch"], best_iou), args)
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint_path))

    if not args.eval_test:
        if args.verbose and args.local_rank == 0:
            print('Starting training loop ...')
        for epoch in range(args.start_epoch, args.epochs):
            train_one_epoch(args, train_dataset, train_dataloader, model,
                            criterion, optimizer, TrainOneBatch, epoch)
            avg_miou = validate_epoch(args, valid_dataloader, model)
            scheduler.step()
            if args.local_rank == 0:
                log(
                    f"Current Validation mIoU at Epoch {epoch} is {avg_miou}", args)
                save_model(args, epoch, model, optimizer, avg_miou,
                           post_fix="_epoch{}{}".format(epoch,args.model_name_postfix))
            if args.local_rank == 0 and avg_miou > best_iou:
                best_iou = avg_miou
                log("*"*50, args)
                log(f"Saving BestIoU Model => {best_iou}", args)
                log("*"*50, args)
                save_model(args, epoch, model, optimizer, best_iou,
                           post_fix=args.model_name_postfix)
    else:
        avg_miou = validate_epoch(args, test_dataloader, model)
        print(f"Testing mIoU is {np.nanmean(avg_miou)}")


def TrainOneBatch(model, opt, data, loss_fun):
    feat = data["feature"].float().cuda()
    gt_segments = data["segments"].float().cuda()
    opt.zero_grad()
    with th.set_grad_enabled(True):
        output1, output2, output3, output4 = model(feat, gt_segments)
        # if args.distributed:
        #     output1 = allgather(output1, args)
        #     output2 = allgather(output2, args)
        #     output3 = allgather(output3, args)
        #     output4 = allgather(output4, args)
        loss = loss_fun((output1, output2, output3, output4), gt_segments)

    loss.backward()
    opt.step()
    dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
    return loss.item()


def compute_iou(c1, c2):
    intersection = max(0, min(c1[1], c2[1])-max(c1[0], c2[0]))
    if intersection == 0:
        return 0
    else:
        union = max(c1[1], c2[1]) - min(c1[0], c2[0])
        return intersection/union

def compute_jacc(c1, c2):
    intersection = max(0, min(c1[1], c2[1])-max(c1[0], c2[0]))
    if intersection == 0:
        return 0
    else:
        union = c1[1]-c1[0]
        return intersection/union


def validate_epoch(args, eval_dataloader, model):
    model.eval()
    running_loss = 0.0
    # valid_bar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader))
    avg_miou = []
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            feat = data["feature"].float().cuda()
            gt_segments = data["segments"].float().cuda()

            _, maxpool_boundaries, pred_seg_index, _ = model(feat, None)
            # print(maxpool_boundaries.size(), pred_seg_index.size())
            # if args.distributed:
            #     maxpool_boundaries = allgather(maxpool_boundaries.unsqueeze(0), args)
            #     pred_seg_index = allgather(pred_seg_index.unsqueeze(0), args)
            # print(maxpool_boundaries.size(), pred_seg_index.size(), gt_segments.size())
            # if args.local_rank == 0:
            l = maxpool_boundaries.size(-1)
            iou_clip = 0.
            n_segments = gt_segments.size(1)
            for i in range(n_segments):
                best_iou = 0.
                for j in range(pred_seg_index.size(0)):
                    if pred_seg_index[j] == model.module.clip_prop_encoding:
                        break
                    column = pred_seg_index[j] % l
                    row = pred_seg_index[j] // l
                    clip_boundary = maxpool_boundaries[0, :, row, column]
                    current_iou = compute_iou(
                        clip_boundary.cpu().numpy(), gt_segments[0, i].cpu().numpy())
                    if current_iou > best_iou:
                        best_iou = current_iou
                iou_clip += best_iou
            avg_miou.append(iou_clip/n_segments)
    sum_iou = torch.empty(1, dtype=torch.float32, device='cuda:{}'.format(args.local_rank)).fill_(np.sum(avg_miou))
    len_iou = torch.empty(1, dtype=torch.float32, device='cuda:{}'.format(args.local_rank)).fill_(len(avg_miou))
    dist.all_reduce(sum_iou, op=torch.distributed.ReduceOp.SUM)
    dist.all_reduce(len_iou, op=torch.distributed.ReduceOp.SUM)
    return sum_iou.item() / len_iou.item()

def train_one_epoch(args, train_dataset, train_dataloader, net, criterion, optimizer, TrainOneBatch, epoch):
    running_loss = 0.0
    net.train()
    train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i_batch, sample_batch in train_bar:
        batch_loss = TrainOneBatch(
            net, optimizer, sample_batch, criterion)
        running_loss += batch_loss
        train_bar.set_description('Training loss: %.4f' %
                                  (running_loss / len(train_dataset)/args.world_size))
    log(f"[{epoch+1} / {args.epochs}] Training loss: {running_loss / len(train_dataset)/args.world_size}", args)


def save_model(args, epoch, model, optimizer, best_iou, post_fix=""):
    """ save current model """

    if th.cuda.device_count() > 1:
        arch = type(model.module).__name__
    else:
        arch = type(model).__name__
    state = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "miou": best_iou,
        "args": args,
    }
    filename = os.path.join(
        args.checkpoint_dir, "checkpoint-epoch{:03d}-iou-{:.4f}.pth.tar".format(
            epoch, best_iou)
    )
    th.save(state, filename)
    os.rename(filename, os.path.join(args.checkpoint_dir,
              "model_best{}.pth.tar".format(post_fix)))


def log(msg, args):
    if args.local_rank == 0:
        print(msg)
        with open(args.log_file_path, "a") as f:
            f.write(msg)
            f.write("\n")


if __name__ == "__main__":
    args = get_args()
    if args.verbose:
        print(args)

    # predefining random initial seeds
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.local_rank==0 and args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
        os.mkdir(args.checkpoint_dir)
    main(args)


# print('*'*100)
# print("Metrics on Test Set")
# print('*'*100)
# best_loss_checkpoint = th.load(os.path.join(
#     args.checkpoint_dir, "model_best_loss.pth.tar"), map_location='cpu')
# net.load_state_dict(best_loss_checkpoint['state_dict'])
# net.cuda()
# net.eval()
# eval(net, test_dataloader)

# net.load_state_dict(best_f1_checkpoint['state_dict'])
# net.cuda()
# net.eval()
# loss, accuracy, f1 = validate_epoch(args, test_dataset, test_dataloader, net, criterion, epoch)
