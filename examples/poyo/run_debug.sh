# 1. multisession, 1.3M, 1000 Epochs
# hippo_multisession.yaml -> hippo_multi_1M_1000ep
python train_debugging.py \
    --config-name train_poyo_mp.yaml \
    log_dir=./logs/hippo_multi_1M_1000ep \
    dataset=hippo_multisession \
    wandb.run_name="hippo_multi_1M_1000ep"

# single - achilles
# hippo_achilles.yaml
python train_debugging.py \
    --config-name train_poyo_mp.yaml \
    log_dir=./logs/hippo_achilles_1M_1000ep \
    dataset=hippo_achilles \
    wandb.run_name="hippo_achilles_1M_1000ep"

# single - buddy
# hippo_buddy.yaml
python train_debugging.py \
    --config-name train_poyo_mp.yaml \
    log_dir=./logs/hippo_buddy_1M_1000ep \
    dataset=hippo_buddy \
    wandb.run_name="hippo_buddy_1M_1000ep"

# single - cicero 
# hippo_cicero.yaml
python train_debugging.py \
    --config-name train_poyo_mp.yaml \
    log_dir=./logs/hippo_cicero_1M_1000ep \
    dataset=hippo_cicero \
    wandb.run_name="hippo_cicero_1M_1000ep"

# single - gatsby 
# hippo_gatsby.yaml
python train_debugging.py \
    --config-name train_poyo_mp.yaml \
    log_dir=./logs/hippo_gatsby_1M_1000ep \
    dataset=hippo_gatsby \
    wandb.run_name="hippo_gatsby_1M_1000ep"
