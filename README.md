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

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Web Demo](#demo)
- [CoreML Export](#coreml)
- [Checkpoints](#checkpoints)
- [iOS App](#ios)
- [License](#license)
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
