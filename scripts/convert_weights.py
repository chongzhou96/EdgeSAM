import torch

import argparse

parser = argparse.ArgumentParser('Convert distilled models to SAM format', add_help=False)
parser.add_argument('src_path', type=str)
parser.add_argument('--encoder-only', action='store_true')
parser.add_argument('--rpn-path', type=str)

args = parser.parse_args()

src_path = args.src_path
tar_path = args.src_path[:-len('.pth')] + '_model.pth'
sam_path = 'weights/sam_vit_h_4b8939.pth'

src_model = torch.load(src_path)['model']
state_dict = dict()
for key in src_model:
    if args.encoder_only:
        new_key = 'image_encoder.'+key
        print(f'From source model: {key} -> {new_key}')
        state_dict[new_key] = src_model[key]
    else:
        print(f'From source model: {key}')
        state_dict[key] = src_model[key]
del src_model

sam = torch.load(sam_path)

for key in sam:
    if key in state_dict:
        continue
    if 'image_encode' in key:
        continue
    state_dict[key] = sam[key]
    print(f'From SAM model: {key}')
del sam

if args.rpn_path is not None:
    rpn_model = torch.load(args.rpn_path)['state_dict']
    for key in rpn_model:
        if key.startswith('neck.'):
            new_key = key.replace('neck', 'fpn')
            print(f'From RPN model: {key} -> {new_key}')
            state_dict[new_key] = rpn_model[key]
        elif key.startswith('bbox_head.'):
            new_key = key.replace('bbox_head', 'rpn_head')
            print(f'From RPN model: {key} -> {new_key}')
            state_dict[new_key] = rpn_model[key]
    del rpn_model
    tar_path = tar_path[:-len('.pth')] + '_rpn.pth'

torch.save(state_dict, tar_path)
print(f'Saved at {tar_path}')