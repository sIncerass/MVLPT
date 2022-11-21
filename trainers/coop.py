import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.samplers import build_sampler
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
from tabulate import tabulate

from trainers.vision_benchmark.evaluation import construct_dataloader, construct_multitask_dataset

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, cfg=None):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.cfg = cfg

    def forward(self, prompts, tokenized_prompts):
        if not self.cfg.TRAINER.CUT_CONTEXTLEN:
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = prompts + self.positional_embedding.type(self.dtype)[:prompts.shape[1], :]
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            for block in self.transformer.resblocks:
                if block.attn_mask.shape[0] != x.shape[0]:
                    block.attn_mask = block.attn_mask[:x.shape[0], :x.shape[0]]
            # x = self.transformer(x)
            from torch.utils.checkpoint import checkpoint_sequential
            act_chunk_size = min(self.cfg.TRAINER.ACT_CKPT, len(self.transformer.resblocks))
            x = checkpoint_sequential(self.transformer.resblocks, act_chunk_size, x) 
            x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        if cfg.TRAINER.CUT_CONTEXTLEN:
            sot_token = _tokenizer.encoder["<|startoftext|>"]
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            max_length = min(clip_model.context_length, max([len([sot_token] + _tokenizer.encode(p) + [eot_token]) for p in prompts]))
        else:
            max_length = clip_model.context_length
        print("Current Context Length is: ", max_length)

        tokenized_prompts = torch.cat([clip.tokenize(p, context_length=max_length) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, dm=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, cfg)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.multi_task_label_pertask = cfg.DATASET.MULTITASK_LABEL_PERTASK
        if self.multi_task_label_pertask:
            self.class_index_pertask_start = torch.arange(dm._num_classes)
            self.class_index_pertask_end = torch.arange(dm._num_classes)
            start_index = 0

            for class_index, task in enumerate(dm._task_names):
                class_num = len(dm._labelmap[task])
                self.class_index_pertask_start[class_index] = start_index
                start_index += class_num
                self.class_index_pertask_end[class_index] = start_index
            self.index = torch.arange(dm._num_classes).unsqueeze(0)

    def forward(self, image, task=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        if self.multi_task_label_pertask:
            # Here we perform prompt selection
            domain_start_indexs = self.class_index_pertask_start[task].unsqueeze(-1)
            domain_end_indexs = self.class_index_pertask_end[task].unsqueeze(-1)
            # print(domain_start_indexs.shape, domain_end_indexs.shape, logits.shape)
            select_index = self.index.repeat(logits.shape[0], 1)
            select_index = (select_index >= domain_start_indexs).float() * (select_index < domain_end_indexs).float()
            # exit()
            logits = logits * select_index.to(logits.device)

        return logits

class CoOpCOOPDataManager(DataManager):

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        # Load dataset
        label_offset = 0
        self.num_classes_list = []
        self.classnames_list = []
        self.lab2cname_list = {}
        self.dataset = None
        self._task_names = cfg.DATASET.DATASET.split(',')
        self._id2task = {}
        self._task_class_idx = {}

        for domain, dataset_name in enumerate(self._task_names):
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            self._id2task[domain] = dataset_name
            dataset = build_dataset(cfg)
            self.num_classes_list.append(dataset._num_classes)
            self.classnames_list += dataset._classnames
            new_lab2cname_dict = {}
            for key, value in dataset._lab2cname.items():
                new_lab2cname_dict[key+label_offset] = value
            self.lab2cname_list.update(new_lab2cname_dict)
            for i in range(len(dataset._train_x)):
                dataset._train_x[i]._label += label_offset
                dataset._train_x[i]._domain = domain
            
            if dataset._train_u:
                for i in range(len(dataset._train_u)):
                    dataset._train_u[i]._label += label_offset
                    dataset._train_u[i]._domain = domain
                if self.dataset is not None:
                    self.dataset._train_u = self.dataset._train_u + dataset._train_u
            if dataset._val:
                for i in range(len(dataset._val)):
                    dataset._val[i]._label += label_offset
                    dataset._val[i]._domain = domain

            for i in range(len(dataset._test)):
                dataset._test[i]._label += label_offset
                dataset._test[i]._domain = domain

            if self.dataset is not None:
                self.dataset._train_x = self.dataset._train_x + dataset._train_x
                self.dataset._test = self.dataset._test + dataset._test
                self.dataset._val = self.dataset._val + dataset._val

            print(dataset._train_u is None, dataset._val is None)
            if self.dataset is None:
                self.dataset = dataset

            self._task_class_idx[dataset_name] = ( label_offset, label_offset + dataset._num_classes )
            label_offset += dataset._num_classes
        dataset = self.dataset
        dataset._classnames = self.classnames_list
        dataset._lab2cname = self.lab2cname_list
        dataset._num_classes = sum(self.num_classes_list)
        print(self.num_classes_list, len(dataset._classnames), dataset._lab2cname, dataset._num_classes)
           
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )
        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

from trainers.vision_benchmark.datasets import class_map_metric, get_metric
import random
class CoOpDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, class_map, train_dataset = construct_dataloader(cfg)

        self._metric = get_metric(class_map_metric[cfg.DATASET.DATASET])
        self._metric_name = class_map_metric[cfg.DATASET.DATASET]
        # Attributes
        self._num_classes = len(class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in enumerate(class_map):
            if isinstance(value, list):
                value = value[0] #random.choice(value)
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass
            # self.show_dataset_summary(cfg)

class CoOpMTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, train_dataset, test_dataloader_by_task = construct_multitask_dataset(cfg)

        self._labelmap = train_dataset.labelmap
        self._task_names = train_dataset._task_names
        self._task2id = { v: k for k, v in enumerate(self._task_names) }
        self._id2task = { k: v for k, v in enumerate(self._task_names) }
        self._metric = { task: get_metric(class_map_metric[task]) for task in self._task_names }
        self._metric_name = { task: class_map_metric[task] for task in self._task_names }
        
        class_idx = 0
        self._task_class_idx = {}
        for task in self._task_names:
            class_num = len(self._labelmap[task])
            self._task_class_idx[task] = ( class_idx, class_idx + class_num )
            class_idx += class_num
        
        from trainers.vision_benchmark.datasets import class_map

        print(self._task_names)
        print(self._labelmap)
        print(class_map.keys())

        mt_class_map = dict()
        for task in self._labelmap:
            for label_idx, label in enumerate(class_map[task]):
                cnt = train_dataset._get_cid( label_idx, task)
                mt_class_map[cnt] = label
        
        print(mt_class_map)
        # Attributes
        self._num_classes = len(mt_class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}

        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in mt_class_map.items():
            if isinstance(value, list):
                value = value[0] #random.choice(value)
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass

@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        if self.cfg.DATASET.COOP:
            classnames = self.dm.dataset.classnames
        else:
            classnames = self.dm.lab2cname.values()
        # classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, dm=self.dm)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        self.multi_task = self.cfg.DATASET.MULTITASK
        self.multi_task_label_pertask = self.cfg.DATASET.MULTITASK_LABEL_PERTASK

        
        if self.cfg.DATASET.COOP:
            dm = CoOpCOOPDataManager(self.cfg)
        elif self.cfg.DATASET.MULTITASK:
            dm = CoOpMTDataManager(self.cfg)
        else:
            dm = CoOpDataManager(self.cfg)
        

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def forward_backward(self, batch):
        image, label, tasks_ = self.parse_batch_train(batch)
        
        # HACK: for multi-label classification, either works
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = label.float()
            # print(label.shape, label.sum(dim=-1).shape)
            label /= label.sum(dim=-1, keepdim=True)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, task=tasks_)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, task=tasks_)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        # HACK: During training, we hack the eval of multi-label by selecting only one class
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = torch.argmax(label, dim=1)
            
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if tasks_ is not None:
            loss_summary.update({"num_tasks": len(set(tasks_.tolist()))})

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

    def parse_batch_test(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

    def model_inference(self, input, task=None):
        return self.model(input, task=task)

    @torch.no_grad()
    def test(self, split=None):
        from tqdm import tqdm
        import copy 
        import numpy as np
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        self.evaluator_task = dict()
        self.elevator_evaluator = { 'y_pred': [], 'y_true': [] }

        if self.multi_task:
            if self.cfg.DATASET.COOP:
                self.evaluator_task = { task: copy.deepcopy( self.evaluator ) for task in self.dm._task_names }
            else:
                self.evaluator_task = { task: copy.deepcopy( self.elevator_evaluator ) for task in self.dm._task_names }
            
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, tasks_ = self.parse_batch_test(batch)
            output = self.model_inference(input, task=tasks_)
            if self.cfg.DATASET.COOP:
                self.evaluator.process(output, label)
            else:
                self.elevator_evaluator['y_pred'].append( output.cpu().detach().numpy() )
                self.elevator_evaluator['y_true'].append( label.cpu().detach().numpy() )

            if tasks_ is not None:
                for out, lab, task in zip(output, label, tasks_):
                    task = self.dm._id2task[task.item()]
                    if self.cfg.DATASET.COOP:
                        class_start, class_end = self.dm._task_class_idx[task]
                        # Evaluate on the task-specific class
                        out = out[class_start:class_end]
                        lab = lab - class_start
                        self.evaluator_task[task].process(out.unsqueeze(0), lab.unsqueeze(0))
                    else:
                        self.evaluator_task[task]['y_pred'].append( [out.cpu().detach().numpy()] )
                        self.evaluator_task[task]['y_true'].append( [lab.cpu().detach().numpy()] )
        
        results_overall = {}
        for task in self.evaluator_task:
            print(f"evaluate on the *{task}* !")
            if self.cfg.DATASET.COOP:
                results = self.evaluator_task[task].evaluate()
                results_overall[task] = results['accuracy']
            else:
                y_true = np.concatenate( self.evaluator_task[task]['y_true'] , axis=0)
                y_pred = np.concatenate( self.evaluator_task[task]['y_pred'] , axis=0)
                class_start, class_end = self.dm._task_class_idx[task]
                y_true = y_true[:, class_start:class_end]
                y_pred = y_pred[:, class_start:class_end]
                
                if self.dm._metric_name[task] == 'accuracy':
                    y_true = np.argmax(y_true, axis=-1)
                metric_result = self.dm._metric[task]( y_true, y_pred )
                results = { self.dm._metric_name[task]: metric_result }
                results_overall[ task ] = metric_result
            print( 'results', results )

            for k, v in results.items():
                tag = f"{split}/{task}/{k}"
                self.write_scalar(tag, v, self.epoch)
        
        print(f"Overall evaluation !")
        if self.multi_task:
            multi_task_evalkey = self.cfg.DATASET.MULTITASK_EVALKEY
            if multi_task_evalkey == 'average':
                results = {'average' : sum([v for k, v in results_overall.items()]) / len(results_overall)}
            else:
                assert multi_task_evalkey in results_overall
                results = {multi_task_evalkey : results_overall[multi_task_evalkey]}
                print(f"select {multi_task_evalkey} as the evaluation key")
        else:
            if not self.cfg.DATASET.COOP:
                y_true = np.concatenate( self.elevator_evaluator['y_true'] , axis=0)
                y_pred = np.concatenate( self.elevator_evaluator['y_pred'] , axis=0)
                results = { self.dm._metric_name: self.dm._metric( y_true, y_pred ) }
            else:
                results = self.evaluator.evaluate()

        print( 'results', results )
        for k, v in results.items():
            tag = f"/{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        return list(results.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
