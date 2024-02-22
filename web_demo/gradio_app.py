# Code credit: [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM).

import torch
import gradio as gr
import numpy as np
from edge_sam import sam_model_registry, SamPredictor
from edge_sam.onnx import SamPredictorONNX
from PIL import ImageDraw
from utils.tools_gradio import fast_process
import copy
import argparse

parser = argparse.ArgumentParser(
    description="Host EdgeSAM as a local web service."
)
parser.add_argument(
    "--checkpoint",
    default="weights/edge_sam_3x.pth",
    type=str,
    help="The path to the PyTorch checkpoint of EdgeSAM."
)
parser.add_argument(
    "--encoder-onnx-path",
    default="weights/edge_sam_3x_encoder.onnx",
    type=str,
    help="The path to the ONNX model of EdgeSAM's encoder."
)
parser.add_argument(
    "--decoder-onnx-path",
    default="weights/edge_sam_3x_decoder.onnx",
    type=str,
    help="The path to the ONNX model of EdgeSAM's decoder."
)
parser.add_argument(
    "--enable-onnx",
    action="store_true",
    help="Use ONNX to speed up the inference.",
)
parser.add_argument(
    "--server-name",
    default="0.0.0.0",
    type=str,
    help="The server address that this demo will be hosted on."
)
parser.add_argument(
    "--port",
    default=8080,
    type=int,
    help="The port that this demo will be hosted on."
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.enable_onnx:
    # device = "cpu"
    predictor = SamPredictorONNX(args.encoder_onnx_path, args.decoder_onnx_path)
else:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bicubic")
    sam = sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)


examples = [
    ["web_demo/assets/1.jpeg"],
    ["web_demo/assets/2.jpeg"],
    ["web_demo/assets/3.jpeg"],
    ["web_demo/assets/4.jpeg"],
    ["web_demo/assets/5.jpeg"],
    ["web_demo/assets/6.jpeg"],
    ["web_demo/assets/7.jpeg"],
    ["web_demo/assets/8.jpeg"],
    ["web_demo/assets/9.jpeg"],
    ["web_demo/assets/10.jpeg"],
    ["web_demo/assets/11.jpeg"],
    ["web_demo/assets/12.jpeg"],
    ["web_demo/assets/13.jpeg"],
    ["web_demo/assets/14.jpeg"],
    ["web_demo/assets/15.jpeg"],
    ["web_demo/assets/16.jpeg"]
]

# Description
title = "<center><strong><font size='8'>EdgeSAM<font></strong> <a href='https://github.com/chongzhou96/EdgeSAM'><font size='6'>[GitHub]</font></a> </center>"

description_p = """ # Instructions for point mode

                1. Upload an image or click one of the provided examples.
                2. Select the point type.
                3. Click once or multiple times on the image to indicate the object of interest.
                4. The Clear button clears all the points.
                5. The Reset button resets both points and the image.

              """

description_b = """ # Instructions for box mode

                1. Upload an image or click one of the provided examples.
                2. Click twice on the image (diagonal points of the box).
                3. The Clear button clears the box.
                4. The Reset button resets both the box and the image.

              """

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def reset(session_state):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []
    session_state['ori_image'] = None
    session_state['image_with_prompt'] = None
    session_state['feature'] = None
    return None, session_state


def reset_all(session_state):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []
    session_state['ori_image'] = None
    session_state['image_with_prompt'] = None
    session_state['feature'] = None
    return None, None, session_state


def clear(session_state):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []
    session_state['image_with_prompt'] = copy.deepcopy(session_state['ori_image'])
    return session_state['ori_image'], session_state


def on_image_upload(
    image,
    session_state,
    input_size=1024
):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    session_state['ori_image'] = copy.deepcopy(image)
    session_state['image_with_prompt'] = copy.deepcopy(image)
    print("Image changed")
    nd_image = np.array(image)
    session_state['feature'] = predictor.set_image(nd_image)

    return image, session_state


def convert_box(xyxy):
    min_x = min(xyxy[0][0], xyxy[1][0])
    max_x = max(xyxy[0][0], xyxy[1][0])
    min_y = min(xyxy[0][1], xyxy[1][1])
    max_y = max(xyxy[0][1], xyxy[1][1])
    xyxy[0][0] = min_x
    xyxy[1][0] = max_x
    xyxy[0][1] = min_y
    xyxy[1][1] = max_y
    return xyxy


def segment_with_points(
    label,
    session_state,
    evt: gr.SelectData,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=False,
):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (97, 217, 54) if label == "Positive" else (237, 34, 13)
    session_state['coord_list'].append([x, y])
    session_state['label_list'].append(1 if label == "Positive" else 0)

    print(f"coord_list: {session_state['coord_list']}")
    print(f"label_list: {session_state['label_list']}")

    draw = ImageDraw.Draw(session_state['image_with_prompt'])
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = session_state['image_with_prompt']

    if args.enable_onnx:
        coord_np = np.array(session_state['coord_list'])[None]
        label_np = np.array(session_state['label_list'])[None]
        masks, scores, _ = predictor.predict(
            features=session_state['feature'],
            point_coords=coord_np,
            point_labels=label_np,
        )
        masks = masks.squeeze(0)
        scores = scores.squeeze(0)
    else:
        coord_np = np.array(session_state['coord_list'])
        label_np = np.array(session_state['label_list'])
        masks, scores, logits = predictor.predict(
            features=session_state['feature'],
            point_coords=coord_np,
            point_labels=label_np,
            num_multimask_outputs=4,
            use_stability_score=True
        )

    print(f'scores: {scores}')
    area = masks.sum(axis=(1, 2))
    print(f'area: {area}')

    annotations = np.expand_dims(masks[scores.argmax()], axis=0)

    seg = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    return seg, session_state


def segment_with_box(
        session_state,
        evt: gr.SelectData,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=False,
):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color, box_outline = 5, (97, 217, 54), 5
    box_color = (0, 255, 0)

    if len(session_state['box_list']) == 0:
        session_state['box_list'].append([x, y])
    elif len(session_state['box_list']) == 1:
        session_state['box_list'].append([x, y])
    elif len(session_state['box_list']) == 2:
        session_state['image_with_prompt'] = copy.deepcopy(session_state['ori_image'])
        session_state['box_list'] = [[x, y]]

    print(f"box_list: {session_state['box_list']}")

    draw = ImageDraw.Draw(session_state['image_with_prompt'])
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = session_state['image_with_prompt']

    if len(session_state['box_list']) == 2:
        box = convert_box(session_state['box_list'])
        xy = (box[0][0], box[0][1], box[1][0], box[1][1])
        draw.rectangle(
            xy,
            outline=box_color,
            width=box_outline
        )

        box_np = np.array(box)
        if args.enable_onnx:
            point_coords = box_np.reshape(2, 2)[None]
            point_labels = np.array([2, 3])[None]
            masks, _, _ = predictor.predict(
                features=session_state['feature'],
                point_coords=point_coords,
                point_labels=point_labels,
            )
            annotations = masks[:, 0, :, :]
        else:
            masks, scores, _ = predictor.predict(
                features=session_state['feature'],
                box=box_np,
                num_multimask_outputs=1,
            )
            annotations = masks

        seg = fast_process(
            annotations=annotations,
            image=image,
            device=device,
            scale=(1024 // input_size),
            better_quality=better_quality,
            mask_random_color=mask_random_color,
            bbox=None,
            use_retina=use_retina,
            withContours=withContours,
        )
        return seg, session_state
    return image, session_state


img_p = gr.Image(label="Input with points", type="pil")
img_b = gr.Image(label="Input with box", type="pil")

with gr.Blocks(css=css, title="EdgeSAM") as demo:
    session_state = gr.State({
        'coord_list': [],
        'label_list': [],
        'box_list': [],
        'ori_image': None,
        'image_with_prompt': None,
        'feature': None
    })

    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode") as tab_p:
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_p.render()
            with gr.Column(scale=1):
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["Positive", "Negative"],
                        value="Positive",
                        label="Point Type"
                    )

                    with gr.Column():
                        clear_btn_p = gr.Button("Clear", variant="secondary")
                        reset_btn_p = gr.Button("Reset", variant="secondary")
                with gr.Row():
                    gr.Markdown(description_p)

        with gr.Row():
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[img_p, session_state],
                    outputs=[img_p, session_state],
                    examples_per_page=8,
                    fn=on_image_upload,
                    run_on_click=True
                )

    with gr.Tab("Box mode") as tab_b:
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_b.render()
            with gr.Row():
                with gr.Column():
                    clear_btn_b = gr.Button("Clear", variant="secondary")
                    reset_btn_b = gr.Button("Reset", variant="secondary")
                    gr.Markdown(description_b)

        with gr.Row():
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[img_b, session_state],
                    outputs=[img_b, session_state],
                    examples_per_page=8,
                    fn=on_image_upload,
                    run_on_click=True
                )

    img_p.upload(on_image_upload, [img_p, session_state], [img_p, session_state])
    img_p.select(segment_with_points, [add_or_remove, session_state], [img_p, session_state])

    clear_btn_p.click(clear, [session_state], [img_p, session_state])
    reset_btn_p.click(reset, [session_state], [img_p, session_state])
    tab_p.select(fn=reset_all, inputs=[session_state], outputs=[img_p, img_b, session_state])

    img_b.upload(on_image_upload, [img_b, session_state], [img_b, session_state])
    img_b.select(segment_with_box, [session_state], [img_b, session_state])

    clear_btn_b.click(clear, [session_state], [img_b, session_state])
    reset_btn_b.click(reset, [session_state], [img_b, session_state])
    tab_b.select(fn=reset_all, inputs=[session_state], outputs=[img_p, img_b, session_state])

demo.queue()
demo.launch(server_name=args.server_name, server_port=args.port)