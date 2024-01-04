# EdgeSAM
**Prompt-In-the-Loop Distillation for On-Device Deployment of SAM**


[Chong Zhou<sup>1</sup>](https://chongzhou96.github.io/), 
[Xiangtai Li<sup>1</sup>](https://lxtgh.github.io/), 
[Chen Change Loy<sup>1*</sup>](https://www.mmlab-ntu.com/person/ccloy/), 
[Bo Dai<sup>2</sup>](https://daibo.info/)

(*corresponding author)

[<sup>1</sup>S-Lab, Nanyang Technological University](https://www.mmlab-ntu.com/), 
[<sup>2</sup>Shanghai Artificial Intelligence Laboratory](https://www.shlab.org.cn/)

[[`Paper`](https://arxiv.org/abs/2312.06660)] 
[[`Project Page`](https://www.mmlab-ntu.com/project/edgesam/)]
[[`Hugging Face Demo`](https://huggingface.co/spaces/chongzhou/EdgeSAM)]
[[`iOS App (TBA)`]()]

https://github.com/chongzhou96/EdgeSAM/assets/15973859/fe1cd104-88dc-4690-a5ea-ff48ae013db3

**Watch the full live demo video: [[YouTube](https://www.youtube.com/watch?v=YYsEQ2vleiE)] [[Bilibili](https://www.bilibili.com/video/BV1294y1P7TC/)]**

## Updates

* **2024/01/01**: EdgeSAM is intergrated into [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling).
* **2023/12/19**: EdgeSAM is now supported in [ISAT](https://github.com/yatengLG/ISAT_with_segment_anything), a segmentation labeling tool.
* **2023/12/16**: EdgeSAM is now supported in [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). Check out the [grounded-edge-sam demo](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/EfficientSAM/grounded_edge_sam.py). Thanks to the IDEA Research team!
* **2023/12/14**: [autodistill-grounded-edgesam](https://github.com/autodistill/autodistill-grounded-edgesam) combines Grounding DINO and EdgeSAM to create Grounded EdgeSAM [[blog](https://blog.roboflow.com/how-to-use-grounded-edgesam/)]. Thanks to the Roboflow team!
* **2023/12/13**: Add ONNX export and speed up the web demo with ONNX as the backend.

## Overview

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
- [CoreML / ONNX Export](#export)
- [Checkpoints](#checkpoints)
- [iOS App](#ios)
- [Acknowledgements](#acknowledgement)
- [Citation](#cite)
- [License](#license)

## Installation <a name="installation"></a>

The code requires `python>=3.8` and we use `torch==2.0.0` and `torchvision==0.15.1`. Please refer to the 
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

1. Clone the repository locally:

```
git clone https://github.com/chongzhou96/EdgeSAM.git && cd EdgeSAM
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
from edge_sam import SamPredictor, sam_model_registry
sam = sam_model_registry["edge_sam"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

Since EdgeSAM follows the same encoder-decoder architecture as SAM, their usages are very similar. One minor difference is that EdgeSAM allows outputting 1, 3, and 4 mask candidates for each prompt, while SAM yields either 1 or 3 masks. For more details, please refer to the [example Jupyter Notebook](https://github.com/chongzhou96/EdgeSAM/blob/master/notebooks/predictor_example.ipynb).

## Web Demo <a name="demo"></a>
After installing EdgeSAM and downloading the checkpoints. You can start an interactive web demo with the following command:

```
python web_demo/gradio_app.py
```

By default, the demo is hosted on `http://0.0.0.0:8080/` and expects `edge_sam_3x.pth` to be stored in the `weights/` folder. You can change the default behavior by:

```
python web_demo/gradio_app.py --checkpoint [CHECKPOINT] --server-name [SERVER_NAME] --port [PORT]
```

Since EdgeSAM can run smoothly on a mobile phone, it's fine if you don't have a GPU.

We've deployed the same web demo in the Hugging Face Space [[link](https://huggingface.co/spaces/chongzhou/EdgeSAM)]. <del> However, since it uses the CPU as the backend and is shared by all users, the experience might not be as good as a local deployment. </del>  Really appreciate the Hugging Face team for supporting us with the GPU!

**Speed up the web demo with ONNX backend**

1. Install the onnxruntime with `pip install onnxruntime` if your machine doesn't have a GPU or `pip install onnxruntime-gpu` if it does (but don't install both of them). Our implementation is tested under version `1.16.3`.

2. Download the ONNX models to the `weights/` folder:

```
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_encoder.onnx
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_decoder.onnx
```

3. Start the demo:

```
python web_demo/gradio_app.py --enable-onnx
```

4. Navigate to http://0.0.0.0:8080 in your browser.

## CoreML / ONNX Export <a name="export"></a>

**CoreML**

We provide a script that can export a trained EdgeSAM PyTorch model to two CoreML model packages, one for the encoder and another for the decoder. You can also download the exported CoreML models at [Checkpoints](#checkpoints).

For encoder:

```
python scripts/export_coreml_model.py [CHECKPOINT]
```

For decoder:

```
python scripts/export_coreml_model.py [CHECKPOINT] --decoder --use-stability-score
```

Since EdgeSAM doesn't perform knowledge distillation on the IoU token of the original SAM, its IoU predictions might not be reliable. Therefore, we use the stability score for mask selection instead. You can stick to the IoU predictions by removing `--use-stability-score`.

The following shows the performance reports of the EdgeSAM CoreML models measured by Xcode on an iPhone 14 (left: encoder, right: decoder):

<p align="center">
  
  ![xcode](https://github.com/chongzhou96/EdgeSAM/assets/15973859/8df54f76-24c9-4ad2-af6d-086b971d073b)
  
</p>

<details>
  <summary> <strong> Known issues and model descriptions </strong> </summary>

  As of `coremltools==7.1`, you may encounter the assertion error during the export, e.g., `assert len(inputs) <= 3 or inputs[3] is None`. One workaround is to comment out this assertion following the traceback path, e.g., `/opt/anaconda3/envs/EdgeSAM/lib/python3.8/site-packages/coremltools/converters/mil/frontend/torch/ops.py line 1573`.

  Since CoreML doesn't support interpolation with dynamic target sizes, the converted CoreML models do not contain the pre-processing, i.e., resize-norm-pad, and the post-processing, i.e., resize back to the original size.

  The encoder takes a `1x3x1024x1024` image as the input and outputs a `1x256x64x64` image embedding. The decoder then takes the image embedding together with point coordinates and point labels as the input. The point coordinates follow the `(height, width)` format with the top-left corner as the `(0, 0)`. The choices of point labels are `0: negative point`, `1: positive point`, `2: top-left corner of box`, and `3: bottom-right corner of box`. 
  
</details>

**ONNX**

Similar to the CoreML export, you can use the following commands to export the encoder and the decoder to ONNX models respectively:

For encoder:

```
python scripts/export_onnx_model.py [CHECKPOINT]
```

For decoder:

```
python scripts/export_onnx_model.py [CHECKPOINT] --decoder --use-stability-score
```

## Checkpoints <a name="checkpoints"></a>

Please download the checkpoints of EdgeSAM from its Hugging Face Space (all the EdgeSAM variants only differ in the number of training images):

| Model               | COCO mAP | PyTorch | CoreML         | ONNX           |
| ------------------- | -------- | ------- | -------------- | -------------- |
| SAM                 | 46.1     | -       | -              | -              |
| EdgeSAM             | 42.1     | [Download](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth) | [[Encoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_encoder.mlpackage.zip)] [[Decoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_decoder.mlpackage.zip)] | [[Encoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_encoder.onnx)] [[Decoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_decoder.onnx)] |
| EdgeSAM-3x          | 42.7     | [Download](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth) | [[Encoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_encoder.mlpackage.zip)] [[Decoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_decoder.mlpackage.zip)] | [[Encoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_encoder.onnx)] [[Decoder](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_decoder.onnx)] |
| EdgeSAM-10x         | 43       | TBA     | TBA            | TBA |

Note: You need to unzip the CoreML model packages before usage.

## iOS App <a name="ios"></a>
We are planning to release the iOS app that we used in the live demo to the App Store. Please stay tuned!

## Acknowledgements <a name="acknowledgement"></a>
This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). We are grateful to [Han Soong Chong](https://www.linkedin.com/in/hansoong-choong-0493a5155/) for his effort in the demonstration application.

We appreciate the following projects, which enable EdgeSAM: [SAM](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), [TinyViT](https://github.com/microsoft/Cream), and [RepViT](https://github.com/THU-MIG/RepViT).

## Citation <a name="cite"></a>
```bibtex
@article{zhou2023edgesam,
  title={EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM},
  author={Zhou, Chong and Li, Xiangtai and Loy, Chen Change and Dai, Bo},
  journal={arXiv preprint arXiv:2312.06660},
  year={2023}
}
```

## License <a name="license"></a>

This project is licensed under <a rel="license" href="https://github.com/chongzhou96/EdgeSAM/blob/master/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.
