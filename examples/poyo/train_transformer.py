import os
os.environ["HYDRA_FULL_ERROR"] = "1"
import logging

import hydra
import lightning as L


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
from torch_brain.models.simple_transformer import TransformerNeuralDecoder
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
)
from torch_brain.transforms import Compose

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


logger = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        modality_spec: ModalitySpec,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.modality_spec = modality_spec
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size  # linear scaling rule

        # special_emb_params = list(self.model.unit_emb.parameters()) + list(
        #     self.model.session_emb.parameters()
        # )

        remaining_params = [
            p
            for n, p in self.model.named_parameters()
            if "unit_emb" not in n and "session_emb" not in n
        ]

        optimizer = SparseLamb(
            [
                # {"params": special_emb_params, "sparse": True},
                {"params": remaining_params},
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

    def training_step(self, batch, batch_idx):
        # mask is always true here 

        # forward pass
        output_values = self.model(**batch["model_inputs"])

        # compute loss

        mask = batch["model_inputs"]["output_mask"]
        # print('mask', mask.shape, mask)
        output_values = output_values[mask]

        # print('target_values before', batch["target_values"].shape)
        target_values = batch["target_values"][mask]
        # print('target_values after', target_values.shape)

        target_weights = batch["target_weights"][mask]
        # print('target_weights', target_weights)


        # print(output_values.shape, target_values.shape, target_weights.shape)
        loss = self.modality_spec.loss_fn(output_values, target_values, target_weights)

        self.log("train_loss", loss, prog_bar=True)

        # Log batch statistics
        # for name in target_values.keys():
        #     preds = torch.cat([pred[name] for pred in output if name in pred])
        #     self.log(f"predictions/mean_{name}", preds.mean())
        #     self.log(f"predictions/std_{name}", preds.std())

        #     targets = target_values[name].float()
        #     self.log(f"targets/mean_{name}", targets.mean())
        #     self.log(f"targets/std_{name}", targets.std())

        # unit_index = batch["model_inputs"]["input_unit_index"].float()
        # self.log("inputs/mean_unit_index", unit_index.mean())
        # self.log("inputs/std_unit_index", unit_index.std())

        return loss

    def validation_step(self, batch, batch_idx):

        # forward pass
        output_values = self.model(**batch["model_inputs"])
        # print('output_values', output_values.shape)

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
        
        temp = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            # session_id_prefix_fn = lambda data: f"hippo1/",
            # unit_id_prefix_fn = lambda data: f"hippo1/",
            # subject_id_prefix_fn = lambda data: f"hippo1/",
        )

        self.num_units: int = len(temp.get_unit_ids())

    def setup_dataset_and_link_model(self, model: TransformerNeuralDecoder):
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

        # self._init_model_vocab(model)

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


    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            # num_workers=self.cfg.num_workers,
            drop_last=False,
            # pin_memory=True,
            # persistent_workers=True, # True if self.cfg.num_workers > 0 else False,
            # prefetch_factor=None, # 2 if self.cfg.num_workers > 0 else None,
        )

        self.log.info(f"Training on {len(train_sampler)} samples")
        self.log.info(f"Training on {len(self.train_dataset.get_unit_ids())} units")
        self.log.info(f"Training on {len(self.get_session_ids())} sessions")

        # for batch in train_loader:
        #     print(batch['model_inputs']['x'].shape)
        #     print(batch['target_values'].shape)
        #     continue

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
            # num_workers=self.cfg.num_workers,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        
        # for batch in val_loader:
        #     print(batch['model_inputs']['x'].shape)
        #     print(batch['target_values'].shape)
        #     continue

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
            # num_workers=self.cfg.num_workers,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")

        return test_loader


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("Simple transformer!")

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

    data_module = DataModule(cfg=cfg)
    
    # make model and data module
    ## Changed for MLP and simple_transformer
    model = hydra.utils.instantiate(
        cfg.model, 
        readout_spec=readout_spec,
        num_units=data_module.num_units,
    )
    # print('model', model)

    data_module.setup_dataset_and_link_model(model)
    # data_module.val_dataloader()

    # Lightning train wrapper
    wrapper = TrainWrapper(
        cfg=cfg,
        model=model,
        modality_spec=readout_spec,
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

    # Test
    trainer.test(wrapper, data_module, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
