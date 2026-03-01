# # 1. multisession, 1.3M, 1000 Epochs
# # hippo_multisession.yaml -> hippo_multi_1M_1000ep
# python train_debugging.py \
#     --config-name train_poyo_mp.yaml \
#     log_dir=./logs/hippo_multi_1M_1000ep \
#     dataset=hippo_multisession \
#     wandb.run_name="hippo_multi_1M_1000ep"

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

#### Your finetuning pipeline is broken now. 
## first run 3/4 training (100 epochs is fine i think)
python train_debugging.py --config-name train_poyo_mp.yaml log_dir=./logs/finetuning_achilles dataset=hippo_not_achilles_10252013_sessinfo wandb.run_name="finetuning_achilles" epochs=100

# Then run the funetunin (unit-identification)
python finetuning.py --config-name train_poyo_mp.yaml log_dir=./logs/finetuning_achilles dataset=hippo_multisession epochs=2 ckpt_path="D:\Pose\Neuro Code\torchbrain\torch_brain\examples\poyo\logs\finetuning\lightning_logs\version_1\checkpoints\epoch=2-step=234.ckpt"

python finetuning.py --config-path "D:\Pose\Neuro Code\torchbrain\torch_brain\examples\poyo\logs\finetuning\lightning_logs\version_1" --config-name "hparams.yaml" +target_session="gatsby_08022013_sessinfo"

