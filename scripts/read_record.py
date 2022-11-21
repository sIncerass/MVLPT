import re
import csv  
import glob
out_name = 'coop_eval_baseline'
# ckpt_folder='/tmp/select_5/CoOp/'
# TRAINER="UPT"
TRAINER="CoOp"
# TRAINER="VPT"


# SHOTS=20
SHOTS=5
# SHOTS=1
CONFIG="vit_b16"
NCTX=4 if TRAINER=="UPT" else 16
# NCTX=4
# eval_cat="IN1K_ADAPT"
# eval_cat="COOP_ADAPT"
# eval_cat="COOP_ADAPT_SEED"
eval_cat="IN1KCOOP_ADAPT_A100"
# eval_cat="IN1KCOOP_ADAPT_A100_SEED"
# eval_cat="IN1KCOOP_ADAPT_ZEROSHOT"
# eval_cat="IN1KCOOP_ADAPT_ZEROSHOT_SEED"

# eval_cat="IN1K_ADAPT_ZERO_SHOT"
# eval_cat="CLIP_ZEROSHOT"
# eval_cat="EVAL_BEST"
# eval_cat="COOP_ADAPT_ZEROSHOT"
# eval_cat="COOP_ADAPT_ZEROSHOT_SEED"
# eval_cat="COOP_ADAPT_A100"
# eval_cat="COOP_ADAPT_A100_SEED"

ckpt_folder=f'/tmp/outputs/COOP_ELEVATER/{TRAINER}/{eval_cat}/'
ckpt_setting=f'/{CONFIG}_{SHOTS}shots/nctx{NCTX}_csc_ctp/'

print(f'{ckpt_folder}/cifar-10/{ckpt_setting}')
seeds = ["1", "2", "3"]
# seeds = ["0"]
if "ZERO" in eval_cat:
    accuracy_index = -1
else:
    accuracy_index = -2
# accuracy_index = -1
# seeds = ["0"]
# out_name = 'vpt_eval'
# ckpt_folder='/tmp/select_5/UPT/'
# ckpt_setting='vit_b16_20shots/nctx16_csc_ctp'
COOP_ELEVATER_DATASET =  ['hateful-memes', 'cifar-10', 'mnist', 'resisc45_clip', 'country211', 'voc-2007-classification', 'cifar-100', 'patch-camelyon', 'rendered-sst2', 'gtsrb', 'fer-2013', 'kitti-distance']

def main():
#     dataset = ['hateful-memes', 'cifar-10', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'resisc45_clip', 'country211', 'food-101', 'stanford-cars', 'fgvc-aircraft-2013b-variants102', 'caltech-101', 'dtd', 'voc-2007-classification', 'cifar-100', 'patch-camelyon', 'rendered-sst2', 'gtsrb', 'eurosat_clip', 'fer-2013', 'kitti-distance']
    with open(f'./scripts/{out_name}.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # dataset = ['hateful-memes', 'cifar-10', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'resisc45_clip', 'country211', 'food-101', 'stanford-cars', 'fgvc-aircraft-2013b-variants102', 'caltech-101', 'dtd', 'voc-2007-classification', 'cifar-100', 'patch-camelyon', 'rendered-sst2', 'gtsrb', 'eurosat_clip', 'fer-2013', 'kitti-distance']
        dataset = COOP_ELEVATER_DATASET
        writer.writerow([" "]+dataset)
        missed = 0
        for seed in seeds:
            temp_row = []
            temp_row.append(f"seed {seed}")
            
            for data1 in dataset:
                # temp_row.append(data1+" seed"+seed)

                # for data2 in dataset:
                #for seed in ["1", "2", "3"]:
                    # with open("/rscratch/shijiayang/Prompt/new0/prompt-moe/CoOp/outputs/evaluation/"+data1+"_"+data2+"/CoOp/vit_b16_20shots/nctx16_cscFalse_ctpmiddle/seed"+seed+"/log.txt") as open_file:
                missed_ = True
                log_files =  glob.glob(f"{ckpt_folder}/{data1}/{ckpt_setting}/seed{seed}/log.txt*")
                
                for log_file in log_files:
                    with open(log_file) as open_file:
                    # with open(f"{ckpt_folder}/{data1}/{ckpt_setting}/seed{seed}/log.txt") as open_file:
                        lines = open_file.readlines()
                        # assert "results" in lines[accuracy_index]
                        number = re.findall('([+-]?[0-9]*\.[0-9]*)', lines[accuracy_index])
                        # print(number, lines[-1])
                        if "results" in lines[accuracy_index] and "test" in lines[accuracy_index-2]:
                            try:
                                temp_row.append(float(number[0]))
                                missed_ = False
                                break
                            except Exception as e:
                                # temp_row.append(" ")
                                continue
                if missed_:
                    temp_row.append(" ")
                    missed += 1
                    print("missed", data1, "seed", seed)
                # break
            writer.writerow(temp_row)
        print(f"okay we missed {missed} entries")


if __name__ == "__main__":
    main()