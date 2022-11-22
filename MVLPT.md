## How to Run

We provide the running scripts in `scripts/mvlpt`, which allow you to reproduce the results on the [MVLPT](https://arxiv.org/pdf/2211.11720.pdf) paper.

Make sure you change the path in `DATA` and run the commands under the main directory `MVLPT/`.

## Multitask Prompt Initialization & Adaptation

All you need is `MVLPT/scripts/mvlpt/main_mt_coopdata_cut.sh` or `MVLPT/scripts/mvlpt/main_mt_elevater_cut.sh`, which contains five input arguments.

`MVLPT/scripts/mvlpt/main_mt_coopdata_cut.sh` uses 11 tasks from [CoOp](https://github.com/KaiyangZhou/CoOp/tree/main/datasets) as source tasks, 

`MVLPT/scripts/mvlpt/main_mt_elevater_cut.sh` uses 20 tasks from [ELEVATER](https://arxiv.org/pdf/2204.08790.pdf) as source tasks (the data for `ELEVATER` will be downloaded automatically), 

You can also adjust the `DATASET` for source tasks. 

`TRAINER` means the TRAINER method (`CoOp, VPT, UPT`) to run different prompt learning methods. 

`CFG` means which config file to use, such as `vit_b16`, `vit_b32` or `vit_l14` (see `MVLPT/configs/trainers/MVLPT/`). Note that in the paper, we use `MVLPT/configs/trainers/MVLPT/vit_b16.yaml` for all settings (please follow the implementation details shown in the paper).

`NCTX` means the context length. 

`SHOTS` means the number of shots. 

`SEED` means the random seed. 

## Single-task Few-Shot Prompt Learning

All you need is `MVLPT/scripts/mvlpt/main_single_elevator_cut.sh`, which contains six input arguments.

`DATASET` takes as input a dataset name, like `cifar-10` or `country211` from [ELEVATER](https://arxiv.org/pdf/2204.08790.pdf) bechmark. The valid names are the files' names in `MVLPT/trainer/vision_benchmark/resources/datasets`.

`MODEL_DIR="--model-dir YOUR_PATH_TO_PRETRAIN_PROMPT_MODEL"` means the directory that contains the pretrained source prompt, it could be gained via `scripts/avg_ckpt.py` to average between your runs with three seeds or from our pretrained [gdrive link](https://drive.google.com/file/d/1YWVLsVcsTEP_z3ehIDgGpFTNalTG_1IE/view?usp=sharing). 

Below we provide examples on how to run [MVLPT](https://arxiv.org/pdf/2211.11720.pdf). 

**MCoOp (M=16)**:
- 1 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh CoOp vit_b16 16 1 1`
- 5 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh CoOp vit_b16 16 5 1`
- 20 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh CoOp vit_b16 16 20 1`
- 5 shot (seed 1) target: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_single_elevater_cut.sh CoOp vit_b16 16 5 1 cifar-10`

**MVPT (M=16)**:
- 1 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh VPT vit_b16 16 1 1`
- 5 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh VPT vit_b16 16 5 1`
- 20 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh VPT vit_b16 16 20 1`
- 5 shot (seed 1) target: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_single_elevater_cut.sh VPT vit_b16 16 5 1 cifar-10`

**MUPT (M=4)**:
- 1 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh UPT vit_b16 4 1 1`
- 5 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh UPT vit_b16 4 5 1`
- 20 shot (seed 1) source: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_mt_coopdata_cut.sh UPT vit_b16 4 20 1`
- 5 shot (seed 1) target: `CUDA_VISIBLE_DEVICES=0 bash scripts/mvlpt/main_single_elevater_cut.sh UPT vit_b16 4 5 1 cifar-10`

After the experiments are finished, you can use `./scripts/read_record.py` to calculate the average results instead of manually looking into the log files. 

**How to evaluate** 

The command is provided in `MVLPT/scripts/mvlpt/main_single_coopdata_cut.sh`. The key arguments are `--model-dir`, `--load-epoch` and `--eval-only`. `--model-dir` indicates the directory where the models are saved (i.e. the entire folder containing `log.txt`, the tensorboard file and `prompt_learner/`). `--load-epoch` tells the code to load the model saved at a specific epoch, like `--load-epoch 50` (see the [source code](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/engine/trainer.py#L169) for more details).


Again, you can use `read_record.py` to automate the calculation of average performance. 

### Zero-Shot CLIP

See `MVLPT/scripts/mvlpt/zeroshot.sh`.

### Linear Probe CLIP

Please move to [lpclip/](lpclip/).