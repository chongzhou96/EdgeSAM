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
  
| Method      | Train Set | COCO AP | COCO AP<sub>s</sub> | COCO AP<sub>m</sub> | COCO AP<sub>l</sub> | GFLops | MParam. | FPS iPhone 14 | FPS 2080 Ti | FPS 3090 |
|-------------|-----------|---------|---------------------|---------------------|---------------------|--------|---------|---------------|-------------|----------|
| SAM         | SA-1B     | 46.1    | 33.6                | 51.9                | 57.7                | 2734.8 | 641.1   | -             | 4.3         | -        |
| FastSAM     | 2% SA-1B  | 37.9    | 23.9                | 43.4                | 50.0                | 887.6  | 68.2    | -             | -           | 25.0*    |
| MobileSAM   | 1% SA-1B  | 39.4    | 26.9                | 44.4                | 52.2                | 38.2   | 9.8     | 4.9           | 103.5       | 100.0*   |
| EdgeSAM     | 1% SA-1B  | 42.2    | 29.6                | 47.6                | 53.9                | 22.1   | 9.6     | 38.7          | 164.3       | -        |
| EdgeSAM-3x  | 3% SA-1B  | 42.7    | 30.0                | 48.6                | 54.5                | 22.1   | 9.6     | 38.7          | 164.3       | -        |
| EdgeSAM-10x | 10% SA-1B | 43.0    | 30.3                | 48.9                | 55.1                | 22.1   | 9.6     | 38.7          | 164.3       | -        |

*In this table, we report the mask mAP on the COCO dataset. ViTDet-H is used as the detector, whose box mAP is 58.7, to provide box prompts. For speed benchmarking, we infer both the encoder and decoder (with a single prompt). FLOPs are calculated based on the 1024x1024 input resolution. Numbers denoted by * are copied from MobileSAM. 3x and 10x represent training with more data. Here, we do not apply an additional mask refinement iteration per the setting of the original SAM paper.*

</details>

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Web Demo](#demo)
- [CoreML Export](#coreml)
- [Checkpoints](#checkpoints)
- [iOS App](#ios)
- [Acknowledgements](#acknowledgement)
- [Citation](#cite)

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
## Web Demo <a name="demo"></a>

## CoreML Export <a name="coreml"></a>

## Checkpoints <a name="checkpoints"></a>

## iOS App <a name="ios"></a>

## Acknowledgements <a name="acknowledgement"></a>
This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). We are grateful to Han Soong Chong for his effort in the demonstration application.

We appreciate the following projects, which enable EdgeSAM: [SAM](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), [TinyViT](https://github.com/microsoft/Cream), and [RepViT](https://github.com/THU-MIG/RepViT).

## Citation <a name="cite"></a>
```bibtex
@article{zhou2023edgesam,
  title={EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM},
  author={Zhou, Chong and Li, Xiangtai and Loy, Chen Change and Dai, Bo},
  journal={arXiv preprint},
  year={2023}
}
```
