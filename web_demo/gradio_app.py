# Code credit: [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM).

import gradio as gr
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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
    help="The path to the EdgeSAM model checkpoint."
)
parser.add_argument(
    "--enable-everything-mode",
    action="store_true",
    help="Since EdgeSAM follows the same encoder-decoder architecture as SAM, the everything mode will infer the "
         "decoder 32x32=1024 times, which is inefficient, thus a longer processing time is expected.",
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
sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bicubic")
sam = sam.to(device=device)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(sam)
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

description_e = """ # Everything mode is NOT recommended.

                Since EdgeSAM follows the same encoder-decoder architecture as SAM, the everything mode will infer the decoder 32x32=1024 times, which is inefficient, thus a longer processing time is expected.

                1. Upload an image or click one of the provided examples.
                2. Click Start to get the segmentation mask.
                3. The Reset button resets the image and masks.

              """

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

global_points = []
global_point_label = []
global_box = []
global_image = None
global_image_with_prompt = None


def reset():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    global_image_with_prompt = None
    return None


def reset_all():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    global_image_with_prompt = None
    if args.enable_everything_mode:
        return None, None, None
    else:
        return None, None


def clear():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image_with_prompt = copy.deepcopy(global_image)
    return global_image


def on_image_upload(image, input_size=1024):
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    global_image = copy.deepcopy(image)
    global_image_with_prompt = copy.deepcopy(image)
    print("Image changed")
    nd_image = np.array(global_image)
    predictor.set_image(nd_image)

    return image


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
        evt: gr.SelectData,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=False,
):
    global global_points
    global global_point_label
    global global_image_with_prompt

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (97, 217, 54) if label == "Positive" else (237, 34, 13)
    global_points.append([x, y])
    global_point_label.append(1 if label == "Positive" else 0)

    print(f'global_points: {global_points}')
    print(f'global_point_label: {global_point_label}')

    draw = ImageDraw.Draw(global_image_with_prompt)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = global_image_with_prompt

    global_points_np = np.array(global_points)
    global_point_label_np = np.array(global_point_label)

    num_multimask_outputs = 4

    masks, scores, logits = predictor.predict(
        point_coords=global_points_np,
        point_labels=global_point_label_np,
        num_multimask_outputs=num_multimask_outputs,
        use_stability_score=True
    )

    print(f'scores: {scores}')
    area = masks.sum(axis=(1, 2))
    print(f'area: {area}')

    if num_multimask_outputs == 1:
        annotations = masks
    else:
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

    return seg


def segment_with_box(
        evt: gr.SelectData,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=False,
):
    global global_box
    global global_image
    global global_image_with_prompt

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color, box_outline = 5, (97, 217, 54), 5
    box_color = (0, 255, 0)

    if len(global_box) == 0:
        global_box.append([x, y])
    elif len(global_box) == 1:
        global_box.append([x, y])
    elif len(global_box) == 2:
        global_image_with_prompt = copy.deepcopy(global_image)
        global_box = [[x, y]]

    print(f'global_box: {global_box}')

    draw = ImageDraw.Draw(global_image_with_prompt)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = global_image_with_prompt

    if len(global_box) == 2:
        global_box = convert_box(global_box)
        xy = (global_box[0][0], global_box[0][1], global_box[1][0], global_box[1][1])
        draw.rectangle(
            xy,
            outline=box_color,
            width=box_outline
        )

        global_box_np = np.array(global_box)

        masks, scores, logits = predictor.predict(
            box=global_box_np,
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
        return seg
    return image


def segment_everything(
        image,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=True,
):
    nd_image = np.array(image)
    masks = mask_generator.generate(nd_image)
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

    return seg


img_p = gr.Image(label="Input with points", type="pil")
img_b = gr.Image(label="Input with box", type="pil")
img_e = gr.Image(label="Input (everything)", type="pil")

if args.enable_everything_mode:
    all_outputs = [img_p, img_b, img_e]
else:
    all_outputs = [img_p, img_b]

with gr.Blocks(css=css, title="EdgeSAM") as demo:
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
                    inputs=[img_p],
                    outputs=[img_p],
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
                    inputs=[img_b],
                    outputs=[img_b],
                    examples_per_page=8,
                    fn=on_image_upload,
                    run_on_click=True
                )

    if args.enable_everything_mode:
        with gr.Tab("Everything mode") as tab_e:
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    img_e.render()
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column():
                            segment_btn_e = gr.Button("Start", variant="primary")
                            reset_btn_e = gr.Button("Reset", variant="secondary")
                            gr.Markdown(description_e)

            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Try some of the examples below ⬇️")
                    gr.Examples(
                        examples=examples,
                        inputs=[img_e],
                        examples_per_page=8,
                    )

    # with gr.Row():
    #     with gr.Column(scale=1):
    #         gr.Markdown(
    #             "<center><img src='https://visitor-badge.laobi.icu/badge?page_id=chongzhou/edgesam' alt='visitors'></center>")

    img_p.upload(on_image_upload, img_p, [img_p])
    img_p.select(segment_with_points, [add_or_remove], img_p)

    clear_btn_p.click(clear, outputs=[img_p])
    reset_btn_p.click(reset, outputs=[img_p])
    tab_p.select(fn=reset_all, outputs=all_outputs)

    img_b.upload(on_image_upload, img_b, [img_b])
    img_b.select(segment_with_box, outputs=[img_b])

    clear_btn_b.click(clear, outputs=[img_b])
    reset_btn_b.click(reset, outputs=[img_b])
    tab_b.select(fn=reset_all, outputs=all_outputs)

    if args.enable_everything_mode:
        segment_btn_e.click(
            segment_everything, inputs=[img_e], outputs=img_e
        )
        reset_btn_e.click(reset, outputs=[img_e])
        tab_e.select(fn=reset_all, outputs=all_outputs)

demo.queue()
demo.launch(server_name=args.server_name, server_port=args.port)
