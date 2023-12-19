import torch
import argparse
from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel
import coremltools as ct
import coremltools.optimize.coreml as cto


parser = argparse.ArgumentParser(
    description="Export the EdgeSAM to an CoreML model."
)

parser.add_argument(
    "checkpoint", type=str, help="The path to the EdgeSAM model checkpoint."
)

parser.add_argument(
    "--quantize",
    action="store_true",
    help="If set, will quantize the model.",
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
)

parser.add_argument(
    "--decoder",
    action="store_true",
    help="If set, export decoder, otherwise export encoder",
)


def export_encoder(sam, args):
    if args.gelu_approximate:
        for n, m in sam.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    sam.forward = sam.forward_dummy_encoder

    traced_model = torch.jit.trace(sam, image_input)
    outputs = [ct.TensorType(name="image_embeddings")]

    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=image_input.shape, name="image")],
        outputs=outputs,
        convert_to="mlprogram"
    )
    return coreml_model


def export_decoder(sam, args):
    sam_decoder = SamCoreMLModel(
        model=sam,
        use_stability_score=args.use_stability_score
    )
    sam_decoder.eval()

    if args.gelu_approximate:
        for n, m in sam_decoder.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    point_coords = torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float)

    image_embeddings_shape = ct.Shape(shape=(1, embed_dim, *embed_size))
    point_coords_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=16, default=1), 2))
    point_labels_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=16, default=1)))

    traced_model = torch.jit.trace(
        sam_decoder,
        [image_embeddings, point_coords, point_labels]
    )

    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=image_embeddings_shape, name="image_embeddings"),
            ct.TensorType(shape=point_coords_shape, name="point_coords"),
            ct.TensorType(shape=point_labels_shape, name="point_labels")
        ],
        outputs=[
            ct.TensorType(name="scores"),
            ct.TensorType(name="masks")
        ],
        convert_to="mlprogram"
    )
    return coreml_model


def run_export(args):
    print("Loading model...")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    basename = args.checkpoint.split('.')[0]

    if args.decoder:
        coreml_model = export_decoder(sam, args)
        out = f'{basename}_decoder.mlpackage'
        quantize_out = f'{basename}_decoder_quant.mlpackage'
    else:
        coreml_model = export_encoder(sam, args)
        out = f'{basename}_encoder.mlpackage'
        quantize_out = f'{basename}_encoder_quant.mlpackage'

    print(f"Exporting CoreML model to {out}...")
    coreml_model.save(out)

    if args.quantize:
        print(f"Quantizing model and writing to {quantize_out}...")
        op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
        config = cto.OptimizationConfig(global_config=op_config)
        compressed_8_bit_model = cto.linear_quantize_weights(coreml_model, config=config)
        compressed_8_bit_model.save(quantize_out)
        print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(args)
