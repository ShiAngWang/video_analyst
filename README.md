#标注说明

本次标注任务针对旷世研究院公布的深度学习训练工程 VideoAnalyst 的单目标跟踪（SOT）部分。该工程基于 pytorch，以算法 SiamFC++ 和 SAT 为项目实例，构建了一套易于任务扩展的深度学习训练/测评框架。

本次标注涵盖工程框架结构与 SiamFC++ 算法重点细节两方面：

* 工程框架架构
  
  该系统整体由 5 个模块构成：
      1. dataloader 模块提供训练数据；
      2. model 模块构建模型结构、初始化和损失函数等，构成模型的整体架构；
      3. optimizer 模块专注于模型（model）的训练细节，比如学习率调节 (lr schedule)、优化器参数 (optim method)、
         梯度调整 (grad modifier) 等；
      4. engine 模块中的 trainer 子模块控制一个epoch的训练流程，tester 子模块控制一个测试集的测试流程；
      5. pipeline 模块构建特定任务的处理流程，可以独立运行测试，或提供对外API接口。
  
  系统配置架构示意图如下：
  
  <br/>
  ![png1](1.png)
  <br/>
  <br/>
  
  实现该系统架构的核心问题是：
  
    1) 如何在这 5 个模块中实现不同的任务；
       
    2) 如何对一个训练或测试流程进行合理的模块化配置。
    
  为解决这两个问题，该工程采用 config 内容与代码模块一一对应的方式，配置即系统构建，整个工程在一套 config 配置树的规划下进行配置与开发；在各个模块内部，
  采用注册器的形式将子模块进行注册以便于检索(对配置树及注册机制的说明参见 [docs/DEVELOP/DEVELOP.md](./docs/DEVELOP/DEVELOP.md))。在此基础上，形成
  了如下的项目整体架构。
  
  ```File Tree
    project_root/
    ├── experiments  # experiment configurations, in yaml format (实验配置，网络的结构配置、数据集配置等等，
                                                                  整个项目都会根据配置文件运行)
    ├── main
    │   ├── train.py  # trainng entry point（训练入口，已经集成化了，当模块构建完成后，
                                             可直接运行 python3 main/train.py or test.py -cfg configfile.yaml）
    │   └── test.py  # test entry point
    ├── video_analyst
    │   ├── data  # modules related to data
    │   │   ├── dataset  # data fetcher of each individual dataset
    │   │   ├── sampler  # data sampler, including inner-dataset and intra-dataset sampling procedure
    │   │   ├── dataloader.py  # data loading procedure
    │   │   └── transformer  # data augmentation
    │   ├── engine  # procedure controller, including traiing control / hp&model loading
    │   │   ├── monitor  # monitor for tasks during training, including visualization / logging / benchmarking
    │   │   ├── trainer.py  # train a epoch
    │   │   ├── tester.py  # test a model on a benchmark
    │   ├── model # model builder
    │   │   ├── backbone  # backbone network builder
    │   │   ├── common_opr  # shared operator (e.g. cross-correlation)
    │   │   ├── task_model  # holistic model builder
    │   │   ├── task_head  # head network builder
    │   │   └── loss  # loss builder
    │   ├── pipeline  # pipeline builder (tracking / vos)
    │   │   ├── segmenter  # segmenter builder for vos
    │   │   ├── tracker  # tracker builder for tracking
    │   │   └── utils  # pipeline utils
    │   ├── config  # configuration manager
    │   ├── evaluation  # benchmark
    │   ├── optim  # optimization-related module (learning rate, gradient clipping, etc.)
    │   │   ├── optimizer # optimizer
    │   │   ├── scheduler # learning rate scheduler
    │   │   └── grad_modifier # gradient-related operation (parameter freezing)
    │   └── utils  # useful tools
    └── README.md
  ```
  
  该工程按照配置即系统构建的设计原则，在配置文件中按照代码模块的实际分布进行相关参数的配置，使其集中化、逻辑化，训练和测试仅需要运行 python3 main/train.py or test.py -cfg configfile.yaml 即可，具体的
  模块构建流程如图。
  
  <br/>
  ![png2](2.png)
  <br/>
  <br/>
  
  由于着重于多任务框架的设计和实现，遵守开闭原则 (Open–closed principle)，因此 video_analyst 具有非常好的多任务扩展属性，可通过添加自定义模块以实现新的工作。(自定义模块
  的说明参见 [docs/TEMPLATES/README.md](./docs/TEMPLATES/README.md))
  

标注参考文档：

[siamfcpp代码讲解 - bilibili](https://www.bilibili.com/video/BV1Hp4y1S74D)

[漫谈视频目标跟踪与分割 - 旷视科技的文章 - 知乎](https://zhuanlan.zhihu.com/p/114743420)

[单目标追踪论文阅读系列（二）——《SiamFC++（AAAI 2020）》 - 零维的文章 - 知乎](https://zhuanlan.zhihu.com/p/93888180)

[video_analyst项目搭建与siamfc++小结 - CSDN](https://blog.csdn.net/PAN_Andy/article/details/105345598)




# Video Analyst

[![Build Status](https://travis-ci.org/MegviiDetection/video_analyst.svg?branch=master)](https://travis-ci.org/MegviiDetection/video_analyst)

This is the implementation of a series of basic algorithms which is useful for video understanding, including Single Object Tracking (SOT), Video Object Segmentation (VOS), etc.

Current implementation list:

* SOT
  * [SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines](https://arxiv.org/abs/1911.06188) [[demo]](https://www.youtube.com/watch?v=TCziWahnXT8&list=PL4KqNq8e6fJkfk35zHRaUd21ExV522JK0&index=4&t=0s&app=desktop)
<div align="center">
  <img src="docs/resources/siamfcpp_ice2.gif" width="800px" />
  <p>Example SiamFC++ outputs.</p>
</div>

* VOS
  * [State-Aware Tracker for Real-Time Video Object Segmentation](https://arxiv.org/abs/2003.00482) 
<div align="center">
  <img src="docs/resources/sat_runman.gif" width="800px" />
  <p>Example SAT outputs.</p>
</div>

## SOT Quick start

### Setup

Please refer to [SETUP.md](docs/TUTORIALS/SETUP.md), [SOT_SETUP.md](docs/TUTORIALS/SOT_SETUP.md)
### Demo

#### SOT video demo
```Bash
# demo with web camera
python3 ./demo/main/video/sot_video.py --config 'experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml' --device cuda --video "webcam" 

# demo with video file, and dump result into video file (optional)
python3 ./demo/main/video/sot_video.py --config 'experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml' --device cuda --video $video_dir/demo.mp4 --output $dump_path/result.mp4

# demo with extracted image files, and dump result into image files (optional)
python3 ./demo/main/video/sot_video.py --config 'experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml' --device cuda --video $video_dir/*.jpg --output $dump_dir
```

### Test

Please refer to [SOT_TEST.md](docs/TUTORIALS/SOT_TEST.md) for detail.

### Training

Please refer to [SOT_TRAINING.md](docs/TUTORIALS/SOT_TRAINING.md) for detail.

## Repository structure (in progress) 项目整体架构

```File Tree
project_root/
├── experiments  # experiment configurations, in yaml format
├── main
│   ├── train.py  # trainng entry point
│   └── test.py  # test entry point
├── video_analyst
│   ├── data  # modules related to data
│   │   ├── dataset  # data fetcher of each individual dataset
│   │   ├── sampler  # data sampler, including inner-dataset and intra-dataset sampling procedure
│   │   ├── dataloader.py  # data loading procedure
│   │   └── transformer  # data augmentation
│   ├── engine  # procedure controller, including traiing control / hp&model loading
│   │   ├── monitor  # monitor for tasks during training, including visualization / logging / benchmarking
│   │   ├── trainer.py  # train a epoch
│   │   ├── tester.py  # test a model on a benchmark
│   ├── model # model builder
│   │   ├── backbone  # backbone network builder
│   │   ├── common_opr  # shared operator (e.g. cross-correlation)
│   │   ├── task_model  # holistic model builder
│   │   ├── task_head  # head network builder
│   │   └── loss  # loss builder
│   ├── pipeline  # pipeline builder (tracking / vos)
│   │   ├── segmenter  # segmenter builder for vos
│   │   ├── tracker  # tracker builder for tracking
│   │   └── utils  # pipeline utils
│   ├── config  # configuration manager
│   ├── evaluation  # benchmark
│   ├── optim  # optimization-related module (learning rate, gradient clipping, etc.)
│   │   ├── optimizer # optimizer
│   │   ├── scheduler # learning rate scheduler
│   │   └── grad_modifier # gradient-related operation (parameter freezing)
│   └── utils  # useful tools
└── README.md
```

## docs

For detail, please refer to markdown files under _docs_.

### SOT 
* [SOT_SETUP.md](docs/TUTORIALS/SOT_SETUP.md): instructions for setting-up
* [SOT_MODEL_ZOO.md](docs/TUTORIALS/SOT_MODEL_ZOO.md): description of released sot models
* [SOT_TRAINING.md](docs/TUTORIALS/SOT_TRAINING.md): details related to training
* [SOT_TEST.md](docs/TUTORIALS/SOT_TEST.md): details related to test
### VOS 
* [VOS_SETUP.md](docs/TUTORIALS/VOS_SETUP.md): instructions for setting-up
* [VOS_MODEL_ZOO.md](docs/TUTORIALS/VOS_MODEL_ZOO.md): description of released sot models
* [VOS_TRAINING.md](docs/TUTORIALS/VOS_TRAINING.md): details related to training
* [VOS_TEST.md](docs/TUTORIALS/VOS_TEST.md): details related to training
### DEVELOP
* [DEVELOP.md](docs/DEVELOP/DEVELOP.md): description of project design (registry, configuration tree, etc.)
* [PIPELINE_API.md](docs/DEVELOP/PIPELINE_API.md): description for pipeline API
* [FORMATTING_INSTRUCTION](docs/DEVELOP/REFORMATTING_INSTRUCTIONS.md): instruction for code formatting (yapf/isort/flake/etc.)

## TODO
[] refine code stype and test cases

## Acknowledgement

* video_analyst/evaluation/vot_benchmark and other related code have been borrowed from [PySOT](https://github.com/STVIR/pysot)
* video_analyst/evaluation/got_benchmark and other related code have been borrowed from [got-toolkit](https://github.com/got-10k/toolkit.git)
* [detectron2](https://github.com/facebookresearch/detectron2)
* [fvcore](https://github.com/facebookresearch/fvcore)
* [pytracking](https://github.com/visionml/pytracking/tree/f0fc646b98166cc28c33f6b12f5169bb6f0647be)
* [DROL](https://github.com/shallowtoil/DROL)

## References

```
@inproceedings{xu2020siamfc++,
  title={SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines.},
  author={Xu, Yinda and Wang, Zeyu and Li, Zuoxin and Yuan, Ye and Yu, Gang},
  booktitle={AAAI},
  pages={12549--12556},
  year={2020}
}
```
```
@inproceedings{chen2020state,
  title={State-Aware Tracker for Real-Time Video Object Segmentation},
  author={Chen, Xi and Li, Zuoxin and Yuan, Ye and Yu, Gang and Shen, Jianxin and Qi, Donglian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9384--9393},
  year={2020}
}
```
## Contact

Maintainer (sorted by family name):

* Xi Chen[@XavierCHEN34](https://github.com/XavierCHEN34)
* Zuoxin Li[@lzx1413](https://github.com/lzx1413)
* Zeyu Wang[@JWarlock](http://github.com/JWarlock)
* Yinda Xu[@MARMOTatZJU](https://github.com/MARMOTatZJU)
