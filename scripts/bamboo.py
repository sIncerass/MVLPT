import json
import random
import os, glob
import shutil
seed=1102
random.seed(seed)

# NSHOTS_TRAIN = 20
# NSHOTS_VAL = 4

NSHOTS_TRAIN = 10
NSHOTS_VAL = 0

NSHOTS_TRAIN_IDEAL = 20 + 4

# gdown 1x53MYBQvRl9Ii3ahYT6chwAfJ48kMFuy
bamboo_json = "/shared/sheng/prompt-moe/CoOp/bamboo_V4.json"
with open(bamboo_json) as f:
    bamboo = json.load(f)
bamboo_id2name = json.load(open("/shared/sheng/bamboo/cls/id_map/id2name.json", "r"))

bamboo_subset_id2name = json.load(open("/shared/sheng/bamboo/shot_10-seed_1102/bamboo_id_map_sample.json", "r"))
bamboo_subset_id2name_sup = dict()

for img_id in bamboo_subset_id2name:
    if img_id not in bamboo['child2father']:
        print(img_id, bamboo_subset_id2name[img_id])
        bamboo_subset_id2name_sup[img_id] = bamboo_subset_id2name[img_id]
    else:
        father_id = bamboo['child2father'][img_id][0]
        bamboo_subset_id2name_sup[ father_id ] = bamboo_id2name[ father_id ]
print(len(bamboo_subset_id2name), len(bamboo_subset_id2name_sup))
print(bamboo['child2father']['n02084071'], [k in bamboo_subset_id2name for k in bamboo['child2father']['n02084071']])

exit()
# bamboo_id2name.update(bamboo['id2name'])
# print(bamboo.keys(), len(bamboo['id2name']))
# print(bamboo['id2name']['n02084071'])
# create dataset
bamboo_data_full_subset = {}

in21k_path = "/shared/group/imagenet_22k/"
places_path = "/shared/group/places/" # train val
inat_path = "/shared/sheng/iNat2021/" # train val
datasets = {}

# for img_id in os.listdir(in21k_path):
#     if img_id not in bamboo_data_full_subset: bamboo_data_full_subset[img_id] = []
#     bamboo_data_full_subset[img_id] += glob.glob(os.path.join(in21k_path, img_id, "*.JPEG"))

# for img_id in bamboo_data_full_subset:
#     for img_path in bamboo_data_full_subset[img_id]:
#         assert os.path.exists(img_path), img_path

# for split in ['train', 'val']:
#     with open(f"/shared/sheng/bamboo/cls/meta/public.{split}.txt", "r") as f:
#         for line in f:
#             # print(line)
#             data = line.strip().split()[0].split("/")[0]
#             datasets[data] = True
#             try:
#                 img_path, img_id = line.strip().split()
#             except:
#                 continue
#             # if "imageNet21K" in line:
#             #     if img_id not in bamboo_data_full_subset: bamboo_data_full_subset[img_id] = []
#             #     img_path = img_path.replace("imageNet21K", in21k_path)
#             #     assert os.path.exists(img_path)
#             #     bamboo_data_full_subset[img_id].append( img_path )
#             if "places365" in line:
#                 if img_id not in bamboo_data_full_subset: bamboo_data_full_subset[img_id] = []
#                 if split == "train":
#                     img_path = img_path.replace("places365", places_path+"train")
#                 else:
#                     img_path = img_path.replace("places365_val", places_path+"val")
                
#                 assert os.path.exists(img_path)
#                 bamboo_data_full_subset[img_id].append( img_path )
            
#             elif "iNat2021" in line:
#                 if split == "val":
#                     if img_id not in bamboo_data_full_subset: bamboo_data_full_subset[img_id] = []
#                     img_path = img_path.replace("iNat2021", f"{inat_path}")
#                     assert os.path.exists(img_path)
#                     bamboo_data_full_subset[img_id].append( img_path )

print(len(datasets), datasets)
print(len(bamboo_data_full_subset))
# with open("/shared/sheng/bamboo/bamboo_data_full_subset_berkeley.json", "w") as f:
#     json.dump(bamboo_data_full_subset, f)

bamboo_data_full_subset = json.load(open("/shared/sheng/bamboo/bamboo_data_full_subset_berkeley.json", "r"))

bamboo_data_sample = dict()
bamboo_id_map_sample = dict()
for img_id in bamboo_data_full_subset:
    if len(bamboo_data_full_subset[img_id]) >= (NSHOTS_TRAIN + NSHOTS_VAL):
        sample_cnt = 0
        random.shuffle(bamboo_data_full_subset[img_id])
        if img_id not in bamboo_data_sample: bamboo_data_sample[img_id] = []
        for img_path in bamboo_data_full_subset[img_id]:
            # if sample_cnt < NSHOTS_TRAIN + NSHOTS_VAL:
            if sample_cnt < NSHOTS_TRAIN_IDEAL:
                # try:
                    os.makedirs(f"/shared/sheng/bamboo/shot_{NSHOTS_TRAIN+NSHOTS_VAL}-seed_{seed}/{img_id}", exist_ok=True)
                    bamboo_path = f"/shared/sheng/bamboo/shot_{NSHOTS_TRAIN+NSHOTS_VAL}-seed_{seed}/{img_id}/{os.path.basename(img_path)}"
                    if not os.path.exists(bamboo_path):
                        shutil.copy(img_path, bamboo_path)
                    bamboo_data_sample[img_id].append(bamboo_path)
                    sample_cnt += 1
                # except Exception as e:
                #     print(e)
                #     print(img_path, sample_cnt)
                #     exit()
                #     continue
            else:
                break
        # bamboo_data_sample[img_id] = random.sample(bamboo_data_full_subset[img_id], NSHOTS_TRAIN + NSHOTS_VAL)
        # assert len(bamboo_data_sample[img_id]) == (NSHOTS_TRAIN + NSHOTS_VAL)
        # os.makedirs(f"/shared/sheng/bamboo/shot_{NSHOTS_TRAIN+NSHOTS_VAL}-seed_{seed}/{img_id}", exist_ok=True)
        # for i, img_path in enumerate(bamboo_data_sample[img_id]):
        #     shutil.copy(img_path, f"/shared/sheng/bamboo/shot_{NSHOTS_TRAIN+NSHOTS_VAL}-seed_{seed}/{img_id}/{img_id}_{i}.JPEG")
        if sample_cnt >= (NSHOTS_TRAIN + NSHOTS_VAL):
            bamboo_id_map_sample[img_id] = bamboo_id2name[img_id]
            print("sucess", img_id, bamboo_id2name[img_id], len(bamboo_data_sample[img_id]), len(bamboo_data_sample))
    else:
        print( "fail", img_id, bamboo_id2name[img_id], len(bamboo_data_full_subset[img_id]) )

print(len(bamboo_id_map_sample))
with open(f"/shared/sheng/bamboo/shot_{NSHOTS_TRAIN+NSHOTS_VAL}-seed_{seed}/bamboo_id_map_sample.json", "w") as f:
    json.dump(bamboo_id_map_sample, f)

print(len(bamboo_data_sample))
with open(f"/shared/sheng/bamboo/shot_{NSHOTS_TRAIN+NSHOTS_VAL}-seed_{seed}/bamboo_data_sample.json", "w") as f:
    json.dump(bamboo_data_sample, f)