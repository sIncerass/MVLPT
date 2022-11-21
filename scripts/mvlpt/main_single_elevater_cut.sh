#!/bin/bash

# custom config
# DATA=/path/to/datasets
#TRAINER=UPT
#TRAINER=VPT
# TRAINER=CoOp
TRAINER=$1

output_dir=~/opensource/ckpt/
#root=/shared/sheng/coop_data
# root=/tmp/ic/
# root=//tmp/coop_data
root=//tmp//coop_data/

# DATASET=$1 # ['hateful-memes', 'cifar-10', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'resisc45_clip', 'country211', 'food-101', 'stanford-cars', 'fgvc-aircraft-2013b-variants102', 'caltech-101', 'dtd', 'voc-2007-classification', 'cifar-100', 'patch-camelyon', 'rendered-sst2', 'gtsrb', 'eurosat_clip', 'fer-2013', 'kitti-distance']
CFG=$2  # config file
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (5, 20, 50)

# PRETRAIN_DATASET="Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101"
PRETRAIN_DATASET="ImageNet,Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101"
# PRETRAIN_DATASET="hateful-memes,cifar-10,mnist,oxford-flower-102,oxford-iiit-pets,resisc45_clip,country211,food-101,stanford-cars,caltech-101,dtd,voc-2007-classification,cifar-100,patch-camelyon,rendered-sst2,gtsrb,eurosat_clip,fer-2013,kitti-distance"
DATASET=$6
MODEL_DIR="--model-dir ${output_dir}/${PRETRAIN_DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp/"
# for SEED in 1 2 3
# for SEED in 1
for SEED in $5
do
    DIR=$output_dir/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    if [ $TRAINER = "UPT" ]; then
        python3 train.py \
        --root $root \
        --seed ${SEED} \
        --trainer MVLPT \
        --config-file configs/trainers/MVLPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --dataset ${DATASET} \
        --shots ${SHOTS} \
        ${MODEL_DIR} \
        TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
        TRAINER.CUT_CONTEXTLEN True
    elif  [ $TRAINER = "VPT" ]; then
        python3 train.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
    else
        python3 train.py \
        --root $root \
        --seed ${SEED} \
        --trainer MVLPT \
        --config-file configs/trainers/MVLPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --dataset ${DATASET} \
        --shots ${SHOTS} \
        ${MODEL_DIR} \
        TRAINER.MVLPT.VPT.N_CTX 0 \
        TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
        TRAINER.CUT_CONTEXTLEN True
    fi
done
