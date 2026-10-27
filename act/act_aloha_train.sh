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
# lerobot-train \
#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --policy.type=act \
#   --policy.push_to_hub=false \
#   --policy.device=cuda \
#   --policy.chunk_size=100 \
#   --policy.n_action_steps=100 \
#   --steps=100 \
#   --batch_size=8 \
#   --save_freq=100 \
#   --log_freq=10 \
#   --wandb.enable=true \
#   --wandb.project=act-aloha-transfer-cube \
#   --wandb.disable_artifact=true \
#   --output_dir=outputs/act_transfer_cube_smoke \
#   --env.type=aloha \
#   --env.task=AlohaTransferCube-v0 \
#   --eval_freq=25 \
#   --eval.n_episodes=2 \
#   --eval.batch_size=1


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
#   --eval.n_episodes=5 \
#   --eval.batch_size=1

  # --dataset.episodes='[0,1,2]' \
  # --dataset.video_backend=pyav \
  

# # with smaller chunk size but n_action_steps == chunk_size
# lerobot-train \
#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --policy.type=act \
#   --policy.push_to_hub=false \
#   --policy.device=cuda \
#   --policy.chunk_size=40 \
#   --policy.n_action_steps=40 \
#   --steps=80000 \
#   --batch_size=8 \
#   --save_freq=80000 \
#   --log_freq=50 \
#   --wandb.enable=true \
#   --wandb.project=act-aloha-transfer-cube \
#   --wandb.disable_artifact=true \
#   --output_dir=outputs/act_transfer_cube_chunk40_80k \
#   --env.type=aloha \
#   --env.task=AlohaTransferCube-v0 \
#   --eval_freq=5000 \
#   --eval.n_episodes=10 \
#   --eval.batch_size=1

#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --dataset.episodes='[0,1,2]' \


# with temporal ensembling. Same as chunk 40 above, only ensembling on.
# n_action_steps must be 1: the policy is queried every step and the overlapping predictions are averaged.
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.chunk_size=40 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --steps=80000 \
  --batch_size=8 \
  --save_freq=80000 \
  --log_freq=50 \
  --wandb.enable=true \
  --wandb.project=act-aloha-transfer-cube \
  --wandb.disable_artifact=true \
  --job_name=chunk40_te \
  --output_dir=outputs/act_transfer_cube_chunk40_te_80k \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval_freq=5000 \
  --eval.n_episodes=10 \
  --eval.batch_size=1
