import json
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models import resnet34


class YC2ProcNetDataset(Dataset):
    """You Cook2 Dataset for Framewise feature extraction."""

    def __init__(self, feature_root=None, data_file=None,
                 dur_file=None, annotation_file=None,
                 split="", frames_per_video=500, max_augs=10, max_samples=None):

        self.split = split
        self.feature_root = feature_root
        data_file_path = os.path.join(feature_root, data_file)
        with open(data_file_path, "r") as f:
            feat_data = json.load(f)

        self.frames_per_video = frames_per_video
        self.dur_frame_dict = {}
        with open(dur_file, "r") as f:
            for line in f:
                vid_name, vid_dur, vid_frame = [
                    l.strip() for l in line.split(',')]
                self.dur_frame_dict[vid_name] = (float(vid_dur), int(vid_frame))

        with open(annotation_file, "r") as f:
            raw_annotation_data = json.load(f)
        raw_data = raw_annotation_data['database']
        self.annotations = {}
        for vid, anns in raw_data.items():
            if anns["subset"] == self.split:
                self.annotations[vid] = anns["annotations"]

        self.data = []
        unique_vid = set()
        for item in feat_data:
            if item["vid"] in self.annotations.keys() and item["aug"] <= max_augs:
                unique_vid.add(item["vid"])
                self.data.append(item)
        if max_samples is not None:
            self.data = self.data[:max_samples]
            print(f"After Max Samples => Samples are {len(self.data)}")
        assert len(unique_vid) == len(self.annotations.keys()
                                     ), "Something is wrong in data count!"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        task = item["task"]
        vid = item["vid"]
        aug_number_str = "{:04d}".format(item["aug"])
        aug_number = item["aug"]
        data_set = "{}_frames".format(item["set"])
        feature_folder_path = os.path.join(
            self.feature_root, data_set, task, vid, aug_number_str)
        feature = np.load(os.path.join(feature_folder_path, "features_resnet.npy"))
        feature = torch.from_numpy(feature)

        duration_t = self.dur_frame_dict[vid][0]
        total_frame = self.dur_frame_dict[vid][1]
        sampling_itv = math.ceil(total_frame/self.frames_per_video)
        time_per_sampled_frame = sampling_itv*duration_t/total_frame
        aug_frame_shift = max(math.floor(
            sampling_itv/10), 1)*self.frames_per_video/total_frame

        temporal_segments = []
        for ann in self.annotations[vid]:
            start = float(ann["segment"][0])
            end = float(ann["segment"][1])
            start = start/time_per_sampled_frame + \
                1-aug_frame_shift*(aug_number-1)
            end = end/time_per_sampled_frame+1-aug_frame_shift*(aug_number-1)
            temporal_segments.append([start, end])

        temporal_segments = torch.from_numpy(np.asarray(temporal_segments))
        return {"feature": feature, "segments": temporal_segments, "task":task, "vid":vid}


if __name__ == '__main__':
    train_dataset = YC2ProcNetDataset(feature_root="/disk/scratch_fast/s2004019/youcook2/raw_videos",
                                      data_file="training_frames.json",
                                      dur_file="/disk/scratch_fast/s2004019/youcook2/yc2/yc2_duration_frame.csv", 
                                      annotation_file="/disk/scratch_fast/s2004019/youcook2/yc2/yc2_new_annotations_trainval_test.json",
                                      split="training", frames_per_video=500, max_augs=2)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True)
    print(len(train_dataloader))
    for _ in range(100):
        for index, data in enumerate(train_dataloader):
            feat = data["feature"]
            gt_segments = data["segments"]
            # print(feat.size(), gt_segments.size())
        
