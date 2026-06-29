import os
os.environ["HYDRA_FULL_ERROR"] = "1"
import logging

import hydra
import lightning as L
import copy


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from temporaldata import Data

from torch_brain.registry import MODALITY_REGISTRY, ModalitySpec
from torch_brain.optim import SparseLamb
from torch_brain.models.poyo import POYO
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import (
    DecodingStitchEvaluator,
    DataForDecodingStitchEvaluator,
)
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
    BalancedRandomFixedWindowSampler
)
from torch_brain.transforms import Compose
import torch.nn.functional as F 
import operator
from numbers import Number
from collections import OrderedDict

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


logger = logging.getLogger(__name__)

_OWN = True # True 
FISH = True 
meta_lr = 0.05 
meta_steps = 5 

def filter_vocab_from_model_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if k not in {'unit_emb.vocab', 'session_emb.vocab'}}, {'unit_emb.vocab', 'session_emb.vocab'}

class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)



def GIP(batch, model, optimizer, loss_fn, n_domains): 
    batch_size = batch['model_inputs']['input_unit_index'].shape[0]
    K = batch_size // n_domains
    assert batch_size % n_domains == 0, 'easiest way'

    grads = {}
    for start_idx in range(0, batch_size, K):
        grads[start_idx] = []
        end_idx = min(start_idx + K, batch_size)
        
        # Create a chunk dictionary by slicing all tensors
        chunk = {
            k: v[start_idx:end_idx] 
            for k, v in batch['model_inputs'].items()
        }
        target_values = batch['target_values'][start_idx:end_idx]
        target_weights = batch['target_weights'][start_idx:end_idx]

        # Pass to model
        output = model(**chunk)
        # compute loss 
        mask = chunk['output_mask']
        output = output[mask]
        target_values = target_values[mask]
        target_weights = target_weights[mask]
        loss = loss_fn(output, target_values, target_weights)
        
        # compute grads 
        loss.backward()
        # get grasds
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in ['unit_emb.weight', 'session_emb.weight', 'token_type_emb.weight', 'latent_emb.weight']:
                    grads[start_idx].append(param.grad.clone().flatten())
        
        optimizer.zero_grad()
        model.zero_grad()

    grad_mat = []
    for domain, domain_grads in grads.items():
        grads[domain] = torch.cat(grads[domain], dim=0)
        grad_mat.append(grads[domain])
    
    grad_mat = torch.stack(grad_mat, dim=0)  # [K, N]
    X = F.normalize(grad_mat, dim=1)
    sim = X @ X.T 
    avg_sim = torch.tril(sim, diagonal=-1).mean()

    return avg_sim.cpu().detach().numpy()

def fish_step(meta_weights, inner_weights, meta_lr):
    _meta_weights, _weights = ParamDict(meta_weights), ParamDict(inner_weights)
    _meta_weights += meta_lr * sum([_weights - _meta_weights], 0 * _meta_weights)
    return _meta_weights

class TrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        modality_spec: ModalitySpec,
        n_domains: int
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.modality_spec = modality_spec
        self.save_hyperparameters(OmegaConf.to_container(cfg))

        self.before_gip = []
        self.after_gip = []
        self.n_domains = n_domains

        self.automatic_optimization = False  # Take control of optimization
        self.gip_optimizer = None

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size  # linear scaling rule

        # special_emb_params = list(self.model.unit_emb.parameters()) + list(
        #     self.model.session_emb.parameters()
        # )
        # remaining_params = [
        #     p
        #     for n, p in self.model.named_parameters()
        #     if "unit_emb" not in n and "session_emb" not in n
        # ]
        # optimizer = SparseLamb(
        #     [
        #         {"params": special_emb_params, "sparse": True},
        #         {"params": remaining_params},
        #     ],
        #     lr=max_lr,
        #     weight_decay=self.cfg.optim.weight_decay,
        # )

        # Store the param grouping function so you can reuse it
        self._get_param_groups = lambda model: [
            {
                "params": list(model.unit_emb.parameters()) + list(model.session_emb.parameters()),
                "sparse": True
            },
            {
                "params": [p for n, p in model.named_parameters() 
                        if "unit_emb" not in n and "session_emb" not in n]
            }
        ]
        # Store optimizer config for reconstruction
        self._optimizer_config = {
            "lr": max_lr,
            "weight_decay": self.cfg.optim.weight_decay,
        }

        self._dummy_param_1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self._dummy_param_2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        optimizer = SparseLamb(
            [
                {
                    "params": [self._dummy_param_1],
                    "sparse": True,
                },
                {
                    "params": [self._dummy_param_2],
                },
            ],
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.lr_decay_start,
            anneal_strategy="cos",
            div_factor=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def on_fit_start(self):
        self.gip_optimizer = SparseLamb(
            self._get_param_groups(self.model),
            **self._optimizer_config
        )

    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        # Reset epoch-level variables
        self.opt_inner_pre = None

    def on_train_batch_start(self, batch, batch_idx):
        """Called right when batch data is available, before training_step.
        This is where you can do your pre-training checks with gradients."""
        # optimizer = self.trainer.optimizers[0]

        self.before_gip.append(GIP(batch, self.model, self.gip_optimizer, self.modality_spec.loss_fn, self.n_domains))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sched = self.lr_schedulers()

        # Your FISH logic here
        # - Deep copy model
        # - Inner loop updates
        # - FISH meta-update
        # Return whatever you want (or None, since you're handling updates manually)

        model_inner = copy.deepcopy(self.model)
        model_inner.train()

        opt_inner = SparseLamb(
            self._get_param_groups(model_inner),
            **self._optimizer_config
        )
        lightning_opt = self.optimizers()

        if self.opt_inner_pre is not None and True: # args.reload_inner_optim:
            opt_inner.load_state_dict(self.opt_inner_pre)
        
        for inner_group, lightning_group in zip(
            opt_inner.param_groups,
            lightning_opt.param_groups
        ):
            inner_group["lr"] = lightning_group["lr"]
            
        # slicing domains
        batch_size = batch['model_inputs']['input_unit_index'].shape[0]
        K = batch_size // self.n_domains
        assert batch_size % self.n_domains == 0, 'easiest way'
        starting_indices = torch.arange(0, batch_size, K)

        # for start_idx in range(0, batch_size, K):
        running_loss = 0.0 

        for idx in torch.randperm(len(starting_indices)):
            start_idx = starting_indices[idx]
            end_idx = min(start_idx + K, batch_size)

            # Create a chunk dictionary by slicing all tensors
            chunk = {
                k: v[start_idx:end_idx] 
                for k, v in batch['model_inputs'].items()
            }
            target_values = batch['target_values'][start_idx:end_idx]
            target_weights = batch['target_weights'][start_idx:end_idx]

            # 
            opt_inner.zero_grad()

            # forward pass
            output_values = model_inner(**chunk)

            # compute loss
            mask = chunk['output_mask']
            output_values = output_values[mask]
            target_values = target_values[mask]
            target_weights = target_weights[mask]
            
            loss = self.modality_spec.loss_fn(output_values, target_values, target_weights)
            self.manual_backward(loss) # loss.backward()
            
            opt_inner.step()

            running_loss += loss 
        
        
        self.log("train_loss", running_loss / len(starting_indices), prog_bar=True)

        self.opt_inner_pre = opt_inner.state_dict()

        # fish update
        model_w, model_leftout_keys = filter_vocab_from_model_state_dict(self.model)
        model_inner_w, model_inner_leftout_keys = filter_vocab_from_model_state_dict(model_inner)

        meta_weights = fish_step(meta_weights=model_w,
                                 inner_weights=model_inner_w,
                                 meta_lr=meta_lr / meta_steps)

        for left_out_key in model_inner_leftout_keys: 
            meta_weights[left_out_key] = self.model.state_dict()[left_out_key]


        self.model.reset_weights(meta_weights)
        
        print("global_step:", self.global_step) # an issue here

        # IMPORTANT:
        # official Lightning bookkeeping step
        dummy_loss = torch.zeros(
            (),
            device=self.device,
            requires_grad=True
        )

        opt.zero_grad()
        self.manual_backward(dummy_loss)
        opt.step()
        sched.step()

        # Return None to tell Lightning you're handling optimization yourself
        return running_loss.detach() / len(starting_indices)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after optimizer step. Model has been updated!
        Perfect place for your post-update checks."""
        
        # optimizer = self.trainer.optimizers[0]

        self.after_gip.append(GIP(batch, self.model, self.gip_optimizer, self.modality_spec.loss_fn, self.n_domains))


    def validation_step(self, batch, batch_idx):

        # forward pass
        output_values = self.model(**batch["model_inputs"])

        # prepare data for evaluator
        # (goes to DecodingStitchEvaluator.on_validation_batch_end)
        data_for_eval = DataForDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output_values,
            targets=batch["target_values"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )

        return data_for_eval

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

    def setup_dataset_and_link_model(self, model: POYO):
        r"""Setup Dataset objects, and update a given model's embedding vocabs (session
        and unit_emb)
        """
        self.sequence_length = model.sequence_length

        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
        
        # print('self.cfg.train_transforms', self.cfg.train_transforms)
        # print('train_transforms', train_transforms) # <torch_brain.transforms.unit_dropout.UnitDropout

        # print('self.cfg.dataset', self.cfg.dataset) # just a dictionary

        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=Compose([*train_transforms, model.tokenize]), # [?] read model tokenize
            # session_id_prefix_fn = lambda data: f"hippo1/",
            # unit_id_prefix_fn = lambda data: f"hippo1/",
            # subject_id_prefix_fn = lambda data: f"hippo1/",

        )
        self.train_dataset.disable_data_leakage_check()

        self._init_model_vocab(model)

        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)

        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="valid",
            transform=Compose([*eval_transforms, model.tokenize]),
            # session_id_prefix_fn = lambda data: f"hippo1/",
            # unit_id_prefix_fn = lambda data: f"hippo1/",
            # subject_id_prefix_fn = lambda data: f"hippo1/",
        )
        self.val_dataset.disable_data_leakage_check()

        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=Compose([*eval_transforms, model.tokenize]),
            # session_id_prefix_fn = lambda data: f"hippo1/",
            # unit_id_prefix_fn = lambda data: f"hippo1/",
            # subject_id_prefix_fn = lambda data: f"hippo1/",
        )
        self.test_dataset.disable_data_leakage_check()


    def _init_model_vocab(self, model: POYO):
        # TODO: Add code for finetuning situation (when model already has a vocab)
        
        model.unit_emb.initialize_vocab(self.get_unit_ids())
        # self.get_unit_ids() -> list of size 9725, its numpy strings, like: 
        # ['perich_miller_population_2018/c_20131003_center_out_reaching/group_electrode_group_M1/elec0/unit_0', 'perich_miller_population_2018/c_20131003_center_out_reaching/group_electrode_group_M1/elec1/unit_1']
        print('self.get_unit_ids()', len(self.get_unit_ids()), (self.get_unit_ids()[0:2]))
        print('Model unit umbeding vocab: ', model.unit_emb, model.unit_emb.weight.shape)

        model.session_emb.initialize_vocab(self.get_session_ids())
        print('self.get_session_ids()', len(self.get_session_ids()), (self.get_session_ids()[0:2]))
        print('Model Sess umbeding vocab: ', model.session_emb, model.session_emb.weight.shape)
        
        # self.get_session_ids(), again list of strings (size 99), like: 
        # ['perich_miller_population_2018/c_20131003_center_out_reaching', 'perich_miller_population_2018/c_20131009_random_target_reaching']

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

    def train_dataloader(self):
        # train_sampler = RandomFixedWindowSampler(
        #     sampling_intervals=self.train_dataset.get_sampling_intervals(),
        #     window_length=self.sequence_length,
        #     generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        # )
        train_sampler = BalancedRandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            batch_size=self.cfg.batch_size, 
            subject_ids=self.train_dataset.get_session_ids(),
            generator=torch.Generator().manual_seed(self.cfg.seed + 1)
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            drop_last=True, # False, for balanced shit
            num_workers=self.cfg.num_workers if not _OWN else 0,
            pin_memory=True if not _OWN else False,
            persistent_workers=True if not _OWN else False, # True if self.cfg.num_workers > 0 else False,
            prefetch_factor=None, # 2 if self.cfg.num_workers > 0 else None,
        )

        self.log.info(f"Training on {len(train_sampler)} samples")
        self.log.info(f"Training on {len(self.train_dataset.get_unit_ids())} units")
        self.log.info(f"Training on {len(self.get_session_ids())} sessions")

        # batch = next(iter(train_loader))
        # print(batch['target_values'].shape)

        # print('train_loader', type(batch))
        # print('batch', type(batch))
        # print('model_inputs', batch['model_inputs'].keys())
        # for k, v in batch['model_inputs'].items():
        #     print(f'model_inputs/{k}', v.shape)
        
        # for k in ['input_timestamps', 'output_timestamps', ]: # 'input_token_type', 
        #     print(f'model_inputs/{k}', batch['model_inputs'][k].min(), batch['model_inputs'][k].max())

        # print('target_values', batch['target_values'].shape)
        # print('target_values', batch['target_values'][0:5])
        # print('target_weights', batch['target_weights'].shape)
        # print('session_id', batch['session_id'])
        # print('absolute_start', batch['absolute_start'])
        # print('eval_mask', batch['eval_mask'].shape)

        # for i, batch in enumerate(train_loader):
        #     print(batch.keys())
        #     print(i, batch['model_inputs']['input_timestamps'].shape) 
        #     # print(batch['model_inputs']['output_timestamps'][:, 0:3], batch['model_inputs']['output_timestamps'][0, -3:])
        #     # print()
        #     break
        # exit()
        
        # print('end of train loader')

        return train_loader

    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        val_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=self.cfg.num_workers if not _OWN else 0,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        
        # for batch in val_loader: 
        #     print(batch['model_inputs']['output_timestamps'][:, 0:3], batch['model_inputs']['output_timestamps'][0, -3:])
        #     print()
        # exit()
        
        return val_loader

    def test_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        test_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=self.cfg.num_workers if not _OWN else 0,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")

        return test_loader

import matplotlib.pyplot as plt 
def plot_two_lists(list1, list2, save_dir: str, name_to_save: str, title1="Plot 1", title2="Plot 2", xlabel="Index", ylabel="Value"):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(10, 8))
    
    # Plot first list
    ax1.plot(list1, 'b-', linewidth=1.5)
    ax1.set_title(title1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(True, alpha=0.3)
    
    # Plot second list
    ax2.plot(list2, 'r-', linewidth=1.5)
    ax2.set_title(title2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_dir}/{name_to_save}')
    plt.clf()
    import numpy as np 

    np.savez(f'{save_dir}/{name_to_save}', before=list1, after=list2)

@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("POYO!")

    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(  
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model, 
        )  

    # get modality details
    # TODO: add test to verify that all recordings have the same readout
    readout_id = cfg.dataset[0].config.readout.readout_id # [?] important. it gets passed down to model, you need to define it for hippocampus
    # print('readout_id', readout_id) # cursor_velocity_2d
    readout_spec = MODALITY_REGISTRY[readout_id] 
    print('readout_spec', readout_spec) # ModalitySpec(id=1, dim=2, type=<DataType.CONTINUOUS: 0>, timestamp_key='cursor.timestamps', value_key='cursor.vel', loss_fn=MSELoss())

    # make model and data module
    model = hydra.utils.instantiate(cfg.model, readout_spec=readout_spec)
    # print('model', model)

    data_module = DataModule(cfg=cfg)
    data_module.setup_dataset_and_link_model(model)

    # Lightning train wrapper
    wrapper = TrainWrapper(
        cfg=cfg,
        model=model,
        modality_spec=readout_spec,
        n_domains = len(data_module.get_session_ids())
    )

    stitch_evaluator = DecodingStitchEvaluator(
        session_ids=data_module.get_session_ids(),
        modality_spec=readout_spec,
    )

    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            save_last=True,
            # save_top_k=1,
            monitor="average_val_metric",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
    ]

    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        limit_val_batches=None,  # Ensure no limit on validation batches
        num_sanity_val_steps=-1 if cfg.sanity_check_validation else 0,
        enable_progress_bar=False,
    )

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)

    ckpt_cb = trainer.checkpoint_callback
    print("best path:", ckpt_cb.best_model_path)
    print("best score:", ckpt_cb.best_model_score)
    print("current score:", ckpt_cb.current_score)
    print("monitor:", ckpt_cb.monitor)

    plot_two_lists(wrapper.before_gip, wrapper.after_gip, f'{cfg.log_dir}', 'FISH', 'before', 'after', )

    # Test
    trainer.test(wrapper, data_module, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
