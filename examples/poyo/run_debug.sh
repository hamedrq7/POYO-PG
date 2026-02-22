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

############### 3. MLPPPPPP 
python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_achilles_MLP_1s_100ep model=mlp.yaml dataset=hippo_achilles wandb.run_name="hippo_achilles_MLP_1s_100ep" epochs=100

python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_buddy_MLP_1s_100ep model=mlp.yaml dataset=hippo_buddy wandb.run_name="hippo_buddy_MLP_1s_100ep" epochs=100

python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_cicero_MLP_1s_100ep model=mlp.yaml dataset=hippo_cicero wandb.run_name="hippo_cicero_MLP_1s_100ep" epochs=100

python train_mlp.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_gatsby_MLP_1s_100ep model=mlp.yaml dataset=hippo_gatsby wandb.run_name="hippo_gatsby_MLP_1s_100ep" epochs=100


################ 4. simple transformer
python train_transformer.py --config-name train_poyo_mp.yaml log_dir=./logs/hippo_achilles_MLP_1s_100ep model=simple_transformer.yaml dataset=hippo_achilles wandb.run_name="hippo_achilles_MLP_1s_100ep" epochs=100