#!/usr/bin/env bash
# ./act/act_aloha.sh
# generic settings

# local setup for smoke test

# export USE_TF=0

# lerobot-train \
#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --policy.type=act \
#   --policy.push_to_hub=false \
#   --policy.device=mps \
#   --policy.chunk_size=100 \
#   --policy.n_action_steps=100 \
#   --steps=100 \
#   --batch_size=8 \
#   --save_freq=100 \
#   --log_freq=10 \
#   --wandb.enable=true \
#   --wandb.project=act-aloha-transfer-cube \
#   --wandb.disable_artifact=true \
#   --dataset.video_backend=pyav \
#   --output_dir=outputs/act_transfer_cube_generic_temp \
#   --env.type=aloha \
#   --env.task=AlohaTransferCube-v0 \
#   --eval_freq=10 \
#   --eval.n_episodes=2 \
#   --eval.batch_size=1

  # --dataset.episodes='[0,1,2]' \

export USE_TF=0
export MUJOCO_GL=egl

# cloud setup smoke test
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --steps=100 \
  --batch_size=8 \
  --save_freq=100 \
  --log_freq=10 \
  --wandb.enable=true \
  --wandb.project=act-aloha-transfer-cube \
  --wandb.disable_artifact=true \
  --output_dir=outputs/act_transfer_cube_smoke \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval_freq=25 \
  --eval.n_episodes=2 \
  --eval.batch_size=1


# cloud setup main run
# lerobot-train \
#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --policy.type=act \
#   --policy.push_to_hub=false \
#   --policy.device=cuda \
#   --policy.chunk_size=100 \
#   --policy.n_action_steps=100 \
#   --steps=100000 \
#   --batch_size=8 \
#   --save_freq=100000 \
#   --log_freq=50 \
#   --wandb.enable=true \
#   --wandb.project=act-aloha-transfer-cube \
#   --wandb.disable_artifact=true \
#   --output_dir=outputs/act_transfer_cube_generic_100k \
#   --env.type=aloha \
#   --env.task=AlohaTransferCube-v0 \
#   --eval_freq=5000 \
#   --eval.n_episodes=10 \
#   --eval.batch_size=1

  # --dataset.episodes='[0,1,2]' \
  # --dataset.video_backend=pyav \
  

# # with smaller chunk size but n_action_steps == chunk_size
# lerobot-train \
#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --dataset.episodes='[0,1,2]' \
#   --policy.type=act \
#   --policy.device=mps \
#   --policy.chunk_size=20 \
#   --policy.n_action_steps=20 \
#   --steps=10000 \
#   --batch_size=8 \
#   --output_dir=outputs/act_insertion_chunk20


# # with temporal_ensemble_coeff enabled.
# # n_action_steps = 20, chunk_size = 100.
# # Burn only 20 of the predicted at a time; the policy is queried every step so the overlapping predictions can be averaged. 
# # 5x the inference cost.
# lerobot-train \
#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --dataset.episodes='[0,1,2]' \
#   --policy.type=act \
#   --policy.device=mps \
#   --policy.chunk_size=100 \
#   --policy.n_action_steps=20 \
#   --policy.temporal_ensemble_coeff=0.01 \
#   --steps=10000 \
#   --batch_size=8 \
#   --output_dir=outputs/act_insertion_ensemble
