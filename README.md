# EdgeSAM
**Prompt-In-the-Loop Distillation for On-Device Deployment of SAM**


[Chong Zhou<sup>1</sup>](https://chongzhou96.github.io/), 
[Xiangtai Li<sup>1</sup>](https://lxtgh.github.io/), 
[Chen Change Loy<sup>1*</sup>](https://www.mmlab-ntu.com/person/ccloy/), 
[Bo Dai<sup>2</sup>](https://daibo.info/)

(*corresponding author)

[<sup>1</sup>S-Lab, Nanyang Technological University](https://www.mmlab-ntu.com/), 
[<sup>2</sup>Shanghai Artificial Intelligence Laboratory](https://www.shlab.org.cn/)

[[`Paper`]()] 
[[`Project Page`](https://mmlab-ntu.github.io/project/edgesam/)]
[[`Hugging Face Demo`](https://huggingface.co/spaces/chongzhou/EdgeSAM)]
[[`iOS App (TBA)`]()]

https://github.com/chongzhou96/EdgeSAM/assets/15973859/fe1cd104-88dc-4690-a5ea-ff48ae013db3

**Watch the full live demo video: [[YouTube](https://www.youtube.com/watch?v=YYsEQ2vleiE)] [[Bilibili]()]**

**EdgeSAM** is an accelerated variant of the Segment Anything Model (SAM), optimized for efficient execution on edge devices with minimal compromise in performance. 
It achieves a **40-fold speed increase** compared to the original SAM, and outperforms MobileSAM, being **14 times as fast** when deployed on edge devices while enhancing the mIoUs on COCO and LVIS by 2.3 and 3.2 respectively. 
EdgeSAM is also the first SAM variant that can run at **over 30 FPS** on an iPhone 14.

<p align="center">
  <img width="900" alt="compare" src="https://github.com/chongzhou96/EdgeSAM/assets/15973859/95a6f308-7300-4cb4-8b1b-b711cdea3f64">
</p>

*In this figure, we show the encoder throughput of EdgeSAM compared with SAM and MobileSAM as well as the mIoU performance on the SA-1K dataset (sampled from SA-1B) with box and point prompts.*

<details>
<summary> <strong>Approach</strong> </summary>
  Our approach involves distilling the original ViT-based SAM image encoder into a purely CNN-based architecture, better suited for edge devices. We carefully benchmark various distillation strategies and demonstrate that task-agnostic encoder distillation fails to capture the full knowledge embodied in SAM. To overcome this bottleneck, we include both the prompt encoder and mask decoder in the distillation process, with box and point prompts in the loop, so that the distilled model can accurately capture the intricate dynamics between user input and mask generation.
  
  <p align="center">
    <img width="612" alt="arch" src="https://github.com/chongzhou96/EdgeSAM/assets/15973859/e706101a-c3d5-4d99-bea5-c6735ce25237">
  </p>
</details>

<details>
<summary> <strong>Performance</strong> </summary>
  
</details>

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Web Demo](#demo)
- [CoreML Export](#coreml)
- [Checkpoints](#checkpoints)
- [iOS App](#ios)
- [Acknowledgement](#Acknowledgement)
- [BibTeX](#cite)

## Installation <a name="installation"></a>

The code requires `python>=3.8` and we use `torch==2.0.0` and `torchvision==0.15.1`. Please refer to the 
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

1. Clone the repository locally:

```
git clone git@github.com:chongzhou96/EdgeSAM.git; cd EdgeSAM
```

2. Install additional dependencies:

```
pip install -r requirements.txt
```

3. Install EdgeSAM:

```
pip install -e .
```

## Usage <a name="usage"></a>

1. Download checkpoints (please refer to [Checkpoints](#checkpoints) for more details about the PyTorch and CoreML checkpoints):

```
mkdir weights
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth
```

2. You can easily incorporate EdgeSAM into your Python code with following lines:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["edge_sam"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```
