# Procedure Segmentation Networks (ProcNets)
This repo hosts the PyTorch re-implementation of work on procedure segmentation of YouCook2 dataset. [Original Source Code](https://github.com/LuoweiZhou/ProcNets-YouCook2)
[Towards Automatic Learning of Procedures from Web Instructional Videos, AAAI 2018](https://arxiv.org/abs/1703.09788)

### Original YouCook2 Dataset
- Training : 1333
- Validation : 457
- Testing : 210 [*only Localization segments, no captions*]


### Available Dataset
*Note*: Original Validation Set is split into current validation and testing splits. 
- Training : 840
- Validation : 183
- Testing : 180

36.392 
### Results
|                   | Validation |      | Testing |         |
| :-----------------|:---------:|:-----:|:---------:|:-----:|
| Method            | Jaccard   | mIoU  | Jaccard   | mIoU  |
| Uniform           | 40.43     | 36.39 | 40.05     | 35.93 |
| ProcNets (LSTM)   | 54.63     | 37.47 | 51.18     | 36.39 |


