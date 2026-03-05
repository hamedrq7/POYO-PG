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
)
from torch_brain.transforms import Compose

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


sequence_length= 1.0 # 1.0
latent_step= 0.125 # 0.125
num_latents_per_step= 16
dim= 64
dim_head= 64
depth= 6
cross_heads= 2
self_heads= 8
ffn_dropout= 0.2
lin_dropout= 0.4
atn_dropout= 0.2
readout_spec = MODALITY_REGISTRY['linear_maze_pos']
path = "D:/Pose/Neuro Code/torchbrain/torch_brain/examples/poyo/logs/hippo_multi_1M_100ep/lightning_logs/version_1/checkpoints/epoch=49-step=500.ckpt"

model = POYO(
    sequence_length=sequence_length,
    readout_spec=readout_spec,
    latent_step=latent_step,
    num_latents_per_step=num_latents_per_step,
    dim=dim,
    depth=depth,
    dim_head=dim_head,
    cross_heads=cross_heads,
    self_heads=self_heads,
    ffn_dropout=ffn_dropout,
    lin_dropout=lin_dropout,
    atn_dropout=atn_dropout
)
model = model.load_pretrained(
    path, 
    readout_spec
)
# for k, v in model.named_parameters():
#     print(k, v.shape)

# print(model.unit_emb.vocab)
sessions = list(model.session_emb.vocab.keys())
sessions.remove('NA')
embeddings = []
for sessId in sessions:
    sessEmbed = model.session_emb.weight[model.session_emb.vocab[sessId]][None, :]
    print(sessId, sessEmbed.shape)
    embeddings.append(sessEmbed)

embeddings = torch.concat(embeddings)
print(embeddings.shape)

# 
units = list(model.unit_emb.vocab.keys())
units.remove('NA')

# First group units 
embeddings = []
session_labels = []
for count, sessId in enumerate(sessions):
    curr_embeddings = []
    for unitId in units: 
        if sessId in unitId:
            temp = model.unit_emb.weight[model.unit_emb.vocab[unitId]][None, :]
            curr_embeddings.append(temp)

    embeddings.append(torch.concat(curr_embeddings))
    session_labels.append(torch.ones((len(curr_embeddings)), dtype=torch.int64)*count)
    # print(sessId, embeddings[-1].shape)

embeddings = torch.concat(embeddings)
session_labels = torch.concat(session_labels)
print(embeddings.shape, session_labels.shape)

# Do PCA see what happens? 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np 

def pca_unit_embeddings(
        embeddings,
        labels,
        save_path: str,
    ):
    model = PCA(n_components=2).fit(embeddings)
    embed_2d = model.transform(embeddings)

    fig, ax = plt.subplots()
    # fig.set_dpi(150)
    # fig.set_size_inches(12, 10)
    c = [
        "#00ff00",
        "#0000ff",
        "#047373",
        "#30CAE5",
        "black",
        "#00ffff",
        "#ffff00",
        "#ff00ff",
        "#009900",
        "#999900",
    ]
    
    marker_size = 16
    list_of_labels = np.unique(labels)
    for class_num in list_of_labels:
        feats_c = embed_2d[labels == class_num]
        print(class_num, feats_c.shape)
        ax.plot(
            feats_c[:, 0],
            feats_c[:, 1],
            ".",
            ms=marker_size,
            c=c[class_num],
            alpha=0.9,
            label=class_num,
        )
        # mean_c = feats_c.mean(0)
        # ax.annotate(f"{class_num}", [mean_c[0], mean_c[1]])
       
    ax.legend()
    ax.set_title(f"Unit Embeddings PCA")
    plt.show()
    # plt.savefig(f"{save_path}/{epoch_num}-2d feats {k}.png")
    plt.clf()


pca_unit_embeddings(embeddings.detach().cpu().numpy(), session_labels.detach().cpu().numpy(), "")
exit()

def pca_session_embeddings(
    embeddings,
    session_ids,
    save_path: str,
):
    model = PCA(n_components=2).fit(embeddings)
    embed_2d = model.transform(embeddings)

    fig, ax = plt.subplots()
    # fig.set_dpi(150)
    # fig.set_size_inches(12, 10)
    c = [
        "black",
        "#0000ff",
        "#990000",
        "#00ffff",
        "#ffff00",
        "#ff00ff",
        "#009900",
        "#999900",
        "#00ff00",
        "#009999",
    ]

    marker_size = 16

    ax.plot(
        embed_2d[:, 0],
        embed_2d[:, 1],
        ".",
        ms=marker_size,
        # c=c[0:len(session_ids)],
        alpha=0.4,
        label=np.array(session_ids),
    )
       
    ax.legend()
    ax.set_title(f"Session Embeddings PCA")
    plt.show()
    # plt.savefig(f"{save_path}/{epoch_num}-2d feats {k}.png")
    plt.clf()

pca_session_embeddings(embeddings.detach().cpu().numpy(), sessions, "")

# Cluster? 
# Cosine Sim? 
# Normalized Euclidean Distance?