import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import random

@DATASET_REGISTRY.register()
class Bamboo(DatasetBase):

    dataset_dir = "bamboo"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        # self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        # self.image_dir = os.path.join(self.dataset_dir, "images")
        
        # self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        # self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        
        # HACK: hack for trevor's group machine dir's
        self.image_dir = root + "/images"
        self.dataset_dir = root
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            # text_file = os.path.join(self.dataset_dir, "classnames.txt")

            # HACK: hack for trevor's group machine dir's
            # text_file = "./scripts/imagenet21k_classnames.txt"
            json_file = root + "/bamboo_id_map_sample.json"
            classnames = self.read_classnames(json_file)

            train, test, _ = self.read_and_split_data(self.image_dir, p_trn=0.8, ignored=[], new_cnames=classnames)
            # train, test, _ = self.read_and_split_data(self.image_dir, p_trn=0.8, ignored=[], new_cnames=classnames)
            # train = self.read_data(classnames, "train")
            # # Follow standard practice to perform evaluation on the val set
            # # Also used as the val set (so evaluate the last-step model)
            # test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        import json
        classnames_origin = json.load(open(text_file, "r"))
        # HACK: make only one classname per class
        for k, v in classnames_origin.items():
            if isinstance(v, list):
                classnames[k] = v[0]
            else:
                classnames[k] = v
        
        # with open(text_file, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip().split(" ")
        #         folder = line[0]
        #         classname = " ".join(line[1:])
        #         classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            # print(n_total, n_train, n_val, len(categories))
            assert n_train > 0 #and n_val > 0 #and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            if n_val > 0:
                val.extend(_collate(images[n_train : n_train + n_val], label, category))
            if n_test > 0:
                test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test

