import torch
import os
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from collections import OrderedDict
import os.path as osp

seeds = [1, 2, 3]

ckpt_dir = "//tmp//Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101/"
# ckpt_path = f"{ckpt_dir}/VPT/vit_b16_5shots/nctx16_csc_ctp/"
ckpt_path = f"{ckpt_dir}/CoOp/vit_b16_5shots/nctx16_csc_ctp/"

ckpt_dir = "prompt_learner/"
ckpt_name = "model-best.pth.tar"
import numpy as np

def average_ckpt(state_dict, ignore=['optimizer', 'scheduler']):
    new_dict = dict()
    print(state_dict['val_result'], state_dict['epoch'])
    for key in state_dict:
        if key in ignore:
            continue
        if isinstance(state_dict[key][0], int):
            new_dict[key] = int(np.average(state_dict[key]))
        elif isinstance(state_dict[key][0], float):
            new_dict[key] = np.average(state_dict[key])
        elif isinstance(state_dict[key][0], dict):
            avg_dict = dict()
            for ckpt_id in range(len(state_dict[key])):
                for param_key in state_dict[key][ckpt_id]:
                    if param_key not in avg_dict:
                        avg_dict[param_key] = []
                    avg_dict[param_key].append( state_dict[key][ckpt_id][param_key] )
            for param_key in avg_dict:
                # print(avg_dict[param_key][0].shape)
                avg_dict[param_key] = torch.stack( avg_dict[param_key] ).mean(dim=0)
                # print(avg_dict[param_key].shape)
            new_dict[key] = dict(avg_dict)
    return new_dict

state = {}
for seed in seeds:
    model_path = f"{ckpt_path}/seed{seed}/{ckpt_dir}{ckpt_name}"
    checkpoint = load_checkpoint(model_path)
    for key in checkpoint:
        if key not in state: state[key] = []
        state[key].append( checkpoint[key] )

avg_ckpt = average_ckpt(state)

print(avg_ckpt.keys())
print(avg_ckpt['val_result'])
print(osp.join(ckpt_path, ckpt_dir))
save_checkpoint(
    {
        "state_dict": avg_ckpt['state_dict'],
        "epoch": avg_ckpt['epoch'],
        "val_result": avg_ckpt['val_result'],
    },
    osp.join(ckpt_path, ckpt_dir),
    is_best=True,
)