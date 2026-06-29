# # 1. multisession, 1.3M, 1000 Epochs
# # hippo_multisession.yaml -> hippo_multi_1M_1000ep
# python train_debugging.py \
#     --config-name train_poyo_mp.yaml \
#     log_dir=./logs/hippo_multi_1M_1000ep \
#     dataset=hippo_multisession \
#     wandb.run_name="hippo_multi_1M_1000ep"

python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/DEBUG dataset=hippo_multisession wandb.run_name="DEBUG"
python train_fish.py --config-name train_poyo_mp.yaml log_dir=./logs/fish-fish dataset=hippo_multisession wandb.run_name="fish-fish"

"""
a x b = 1.0 : 
                                 metric     value
0  rat_hippo/achilles_10252013_sessinfo  0.922155
1     rat_hippo/buddy_06272013_sessinfo  0.902213
2    rat_hippo/cicero_09012014_sessinfo  0.460629
3    rat_hippo/gatsby_08022013_sessinfo  0.682905
4                   average_test_metric  0.741975
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃             Test metric              ┃             DataLoader 0             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         average_test_metric          │          0.7419754266738892          │
│ rat_hippo/achilles_10252013_sessinfo │          0.9221545457839966          │
│  rat_hippo/buddy_06272013_sessinfo   │          0.9022129774093628          │
│  rat_hippo/cicero_09012014_sessinfo  │          0.4606291651725769          │
│  rat_hippo/gatsby_08022013_sessinfo  │          0.6829048991203308          │
└──────────────────────────────────────┴──────────────────────────────────────┘
--------------------------------------------------------------------------------
meta_lr = 0.5 
meta_steps = 5 
                                 metric     value
0  rat_hippo/achilles_10252013_sessinfo  0.933576
1     rat_hippo/buddy_06272013_sessinfo  0.892914
2    rat_hippo/cicero_09012014_sessinfo  0.476920
3    rat_hippo/gatsby_08022013_sessinfo  0.737332
4                   average_test_metric  0.760185
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃             Test metric              ┃             DataLoader 0             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         average_test_metric          │          0.7601853609085083          │
│ rat_hippo/achilles_10252013_sessinfo │          0.9335759878158569          │
│  rat_hippo/buddy_06272013_sessinfo   │          0.8929136395454407          │
│  rat_hippo/cicero_09012014_sessinfo  │          0.4769202470779419          │
│  rat_hippo/gatsby_08022013_sessinfo  │          0.7373315095901489          │
└──────────────────────────────────────┴──────────────────────────────────────-
--------------------------------------------------------------------------------

"""

# # single - achilles
# # hippo_achilles.yaml
# python train_debugging.py \
#     --config-name train_poyo_mp.yaml \
#     log_dir=./logs/hippo_achilles_1M_1000ep \
#     dataset=hippo_achilles \
#     wandb.run_name="hippo_achilles_1M_1000ep"

# # single - buddy
# # hippo_buddy.yaml
# python train_debugging.py \
#     --config-name train_poyo_mp.yaml \
#     log_dir=./logs/hippo_buddy_1M_1000ep \
#     dataset=hippo_buddy \
#     wandb.run_name="hippo_buddy_1M_1000ep"

# # single - cicero 
# # hippo_cicero.yaml
# python train_debugging.py \
#     --config-name train_poyo_mp.yaml \
#     log_dir=./logs/hippo_cicero_1M_1000ep \
#     dataset=hippo_cicero \
#     wandb.run_name="hippo_cicero_1M_1000ep"

# # single - gatsby 
# # hippo_gatsby.yaml
# python train_debugging.py \
#     --config-name train_poyo_mp.yaml \
#     log_dir=./logs/hippo_gatsby_1M_1000ep \
#     dataset=hippo_gatsby \
#     wandb.run_name="hippo_gatsby_1M_1000ep"

############### scrambled | 2. multisession, 1.3M, 1000 Epochs --> scrambled
# # hippo_multisession.yaml -> hippo_multi_1M_1000ep
# python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/scrambled_hippo_multi_1M_100ep  dataset=hippo_multisession wandb.run_name="scrambled_hippo_multi_1M_100ep"

# ############### 3. MLPPPPPP 
# python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_achilles_MLP_1s_100ep model=mlp.yaml dataset=hippo_achilles wandb.run_name="hippo_achilles_MLP_1s_100ep" epochs=100

# python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_buddy_MLP_1s_100ep model=mlp.yaml dataset=hippo_buddy wandb.run_name="hippo_buddy_MLP_1s_100ep" epochs=100

# python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_cicero_MLP_1s_100ep model=mlp.yaml dataset=hippo_cicero wandb.run_name="hippo_cicero_MLP_1s_100ep" epochs=100

# python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_gatsby_MLP_1s_100ep model=mlp.yaml dataset=hippo_gatsby wandb.run_name="hippo_gatsby_MLP_1s_100ep" epochs=100


# ################ 4. simple transformer
# python train_transformer.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_achilles_simpleTrans_1s_500ep model=simple_transformer.yaml dataset=hippo_achilles wandb.run_name="hippo_achilles_simpleTrans_1s_500ep" epochs=500

# python train_transformer.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_buddy_simpleTrans_1s_500ep model=simple_transformer.yaml dataset=hippo_buddy wandb.run_name="hippo_buddy_simpleTrans_1s_500ep" epochs=500

# python train_transformer.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_cicero_simpleTrans_1s_500ep model=simple_transformer.yaml dataset=hippo_cicero wandb.run_name="hippo_cicero_simpleTrans_1s_500ep" epochs=500

# python train_transformer.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_gatsby_simpleTrans_1s_500ep model=simple_transformer.yaml dataset=hippo_gatsby wandb.run_name="hippo_gatsby_simpleTrans_1s_500ep" epochs=500


# # Debugging (testing tokenizer...)
# python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/finetuning dataset=hippo_multisession epochs=3 

############### Your finetuning pipeline is broken now. 
# First run a 100 epochs all sessions
# python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_multi_1M_100ep dataset=hippo_multisession wandb.run_name="hippo_multi_1M_100ep" epochs=100

# ## first run 3/4 training (100 epochs is fine i think)
# python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/pretraining_for_achilles dataset=hippo_not_achilles_10252013_sessinfo wandb.run_name="pretraining_achilles" epochs=100

# python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/pretraining_for_buddy dataset=hippo_not_buddy_06272013_sessinfo wandb.run_name="pretraining_buddy" epochs=100

# python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/pretraining_for_cicero dataset=hippo_not_cicero_09012014_sessinfo wandb.run_name="pretraining_cicero" epochs=100

# python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/pretraining_for_gatsby dataset=hippo_not_gatsby_08022013_sessinfo wandb.run_name="pretraining_gatsby" epochs=100

# python finetuning.py --config-name train_poyo_mp.yaml epochs=100 log_dir=./logs/finetuning_achilles wandb.run_name=finetuning_achilles +target_session=achilles_10252013_sessinfo +pretrained_model="D:\Pose\Neuro Code\torchbrain\torch_brain\examples\poyo\logs\pretraining_for_achilles\lightning_logs\version_0\checkpoints\best.ckpt"

# python finetuning.py --config-name train_poyo_mp.yaml epochs=100 log_dir=./logs/finetuning_buddy wandb.run_name=finetuning_buddy +target_session=buddy_06272013_sessinfo +pretrained_model="D:\Pose\Neuro Code\torchbrain\torch_brain\examples\poyo\logs\pretraining_for_buddy\lightning_logs\version_0\checkpoints\best.ckpt"

# python finetuning.py --config-name train_poyo_mp.yaml epochs=100 log_dir=./logs/finetuning_cicero wandb.run_name=finetuning_cicero +target_session=cicero_09012014_sessinfo +pretrained_model="D:\Pose\Neuro Code\torchbrain\torch_brain\examples\poyo\logs\pretraining_for_cicero\lightning_logs\version_0\checkpoints\best.ckpt"

# python finetuning.py --config-name train_poyo_mp.yaml epochs=100 log_dir=./logs/finetuning_gatsby wandb.run_name=finetuning_gatsby +target_session=gatsby_08022013_sessinfo +pretrained_model="D:\Pose\Neuro Code\torchbrain\torch_brain\examples\poyo\logs\pretraining_for_gatsby\lightning_logs\version_0\checkpoints\best.ckpt"

# # Change the yaml file in XXXX and pass the model path to it... 
# # Then run the funetuning (unit-identification)
# python finetuning.py --config-path "/content/POYO-PG/examples/poyo/logs/finetuning_achilles/wandb/run-20260301_010157-54iyeusz/files" --config-name "config.yaml" +target_session="achilles_10252013_sessinfo" wandb.run_name="finetuning_achilles" +pretrained_model="/content/POYO-PG/examples/poyo/logs/finetuning_achilles/poyo_hippo_benchmarking/5zzdi9ln/checkpoints/epoch=79-step=720.ckpt" epochs=100 

# python neuron_probing.py --config-path "D:\Pose\Neuro Code\torchbrain\torch_brain\examples\poyo\logs\hippo_multi_1M_100ep\lightning_logs\version_1" --config-name "hparams.yaml"