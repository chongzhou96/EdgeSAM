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

# Description
title = "<center><strong><font size='8'>EdgeSAM<font></strong></center>"

description_p = """ # Instructions for point mode

                1. Upload an image or click one of the provided examples.
                2. Select the point type.
                3. Click once or multiple times on the image to indicate the object of interest.
                4. Click Start to get the segmentation mask.
                5. The clear button clears all the points.
                6. The reset button resets both points and the image.

              """

description_b = """ # Instructions for box mode

                1. Upload an image or click one of the provided examples.
                2. Click twice on the image (diagonal points of the box).
                3. Click Start to get the segmentation mask.
                4. The clear button clears the box.
                5. The reset button resets both the box and the image.

              """

description_e = """ # Everything mode is NOT recommended.

                Since EdgeSAM follows the same encoder-decoder architecture as SAM, the everything mode will infer the decoder 32x32=1024 times, which is inefficient, thus a longer processing time is expected.

              """

examples = [
    ["web_demo/assets/picture1.jpg"],
    ["web_demo/assets/picture2.jpg"],
    ["web_demo/assets/picture3.jpg"],
    ["web_demo/assets/picture4.jpg"],
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

global_points = []
global_point_label = []
global_box = []
global_image = None


def reset():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    return None, None


def reset_all():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    if args.enable_everything_mode:
        return None, None, None, None, None, None
    else:
        return None, None, None, None


def clear():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global_points = []
    global_point_label = []
    global_box = []
    return global_image, None


def on_image_upload(image, input_size=1024):
    global global_points
    global global_point_label
    global global_box
    global global_image
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
    print("Image changed")
    nd_image = np.array(global_image)
    predictor.set_image(nd_image)

    return image, None


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


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label
    # global global_image

    x, y = evt.index[0], evt.index[1]
    # x = int(x * scale)
    # y = int(y * scale)
    point_radius, point_color = 10, (97, 217, 54) if label == "Positive" else (237, 34, 13)
    global_points.append([x, y])
    global_point_label.append(1 if label == "Positive" else 0)

    print(f'global_points: {global_points}')
    print(f'global_point_label: {global_point_label}')

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image


def get_box_with_draw(image, evt: gr.SelectData):
    global global_box
    # global global_image

    x, y = evt.index[0], evt.index[1]
    # x = float(x * scale)
    # y = float(y * scale)
    point_radius, point_color, box_outline = 5, (97, 217, 54), 5
    box_color = (0, 255, 0)

    if len(global_box) == 0:
        global_box.append([x, y])
    elif len(global_box) == 1:
        global_box.append([x, y])
    elif len(global_box) == 2:
        global_box = [[x, y]]

    print(f'global_box: {global_box}')

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )

    if len(global_box) == 2:
        global_box = convert_box(global_box)
        xy = (global_box[0][0], global_box[0][1], global_box[1][0], global_box[1][1])
        draw.rectangle(
            xy,
            outline=box_color,
            width=box_outline
        )

    return image


def segment_with_points(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=False,
):
    global global_points
    global global_point_label

    global_points_np = np.array(global_points)
    global_point_label_np = np.array(global_point_label)

    if global_points_np.size == 0 and global_point_label_np.size == 0:
        print("No point selected")
        return image, image

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

    return image, seg


def segment_with_box(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=False,
):
    global global_box
    global_box_np = np.array(global_box)

    if global_box_np.size < 4:
        print("No box selected")
        return image, image

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

    return image, seg


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


cond_img_p = gr.Image(label="Input with points", type="pil")
cond_img_b = gr.Image(label="Input with box", type="pil")
cond_img_e = gr.Image(label="Input (everything)", type="pil")

segm_img_p = gr.Image(label="Segmented Image with points", interactive=False, type="pil")
segm_img_b = gr.Image(label="Segmented Image with box", interactive=False, type="pil")
segm_img_e = gr.Image(label="Segmented Everything", interactive=False, type="pil")

if args.enable_everything_mode:
    all_outputs = [cond_img_p, cond_img_b, cond_img_e, segm_img_p, segm_img_b, segm_img_e]
else:
    all_outputs = [cond_img_p, cond_img_b, segm_img_p, segm_img_b]

with gr.Blocks(css=css, title="EdgeSAM") as demo:

    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode") as tab_p:
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["Positive", "Negative"],
                        value="Positive",
                        label="Point Type"
                    )

                    with gr.Column():
                        segment_btn_p = gr.Button(
                            "Start", variant="primary"
                        )
                        clear_btn_p = gr.Button("Clear", variant="secondary")
                        reset_btn_p = gr.Button("Reset", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    outputs=[cond_img_p, segm_img_p],
                    examples_per_page=4,
                    fn=on_image_upload,
                    run_on_click=True
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    with gr.Tab("Box mode") as tab_b:
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_b.render()

            with gr.Column(scale=1):
                segm_img_b.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        segment_btn_b = gr.Button(
                            "Start", variant="primary"
                        )
                        clear_btn_b = gr.Button("Clear", variant="secondary")
                        reset_btn_b = gr.Button("Reset", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_b],
                    outputs=[cond_img_b, segm_img_b],
                    examples_per_page=4,
                    fn=on_image_upload,
                    run_on_click=True
                )

            with gr.Column():
                # Description
                gr.Markdown(description_b)

    if args.enable_everything_mode:
        with gr.Tab("Everything mode") as tab_e:
            # Images
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    cond_img_e.render()

                with gr.Column(scale=1):
                    segm_img_e.render()

            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            segment_btn_e = gr.Button(
                                "Start", variant="primary"
                            )
                            reset_btn_e = gr.Button("Reset", variant="secondary")

                    gr.Markdown("Try some of the examples below ⬇️")
                    gr.Examples(
                        examples=examples,
                        inputs=[cond_img_e],
                        examples_per_page=4,
                    )

                with gr.Column():
                    # Description
                    gr.Markdown(description_e)

    cond_img_p.upload(on_image_upload, cond_img_p, [cond_img_p, segm_img_p])
    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)
    segment_btn_p.click(
        segment_with_points, inputs=[cond_img_p], outputs=[cond_img_p, segm_img_p]
    )
    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
    reset_btn_p.click(reset, outputs=[cond_img_p, segm_img_p])
    tab_p.select(fn=reset_all, outputs=all_outputs)

    cond_img_b.select(get_box_with_draw, [cond_img_b], cond_img_b)
    segment_btn_b.click(
        segment_with_box, inputs=[cond_img_b], outputs=[cond_img_b, segm_img_b]
    )
    clear_btn_b.click(clear, outputs=[cond_img_b, segm_img_b])
    reset_btn_b.click(reset, outputs=[cond_img_b, segm_img_b])
    tab_b.select(fn=reset_all, outputs=all_outputs)

    if args.enable_everything_mode:
        segment_btn_e.click(
            segment_everything, inputs=[cond_img_e], outputs=segm_img_e
        )
        reset_btn_e.click(reset, outputs=[cond_img_e, segm_img_e])
        tab_e.select(fn=reset_all, outputs=all_outputs)

demo.queue()
demo.launch(server_name=args.server_name, server_port=args.port)
