# generic settings
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.episodes='[0,1,2]' \
  --policy.type=act \
  --policy.device=mps \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --steps=10000 \
  --batch_size=8 \
  --output_dir=outputs/act_insertion_generic
  

# with smaller chunk size but n_action_steps == chunk_size
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.episodes='[0,1,2]' \
  --policy.type=act \
  --policy.device=mps \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --steps=10000 \
  --batch_size=8 \
  --output_dir=outputs/act_insertion_chunk20


# with temporal_ensemble_coeff enabled.
# n_action_steps = 20, chunk_size = 100.
# Burn only 20 of the predicted at a time; the policy is queried every step so the overlapping predictions can be averaged. 
# 5x the inference cost.
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.episodes='[0,1,2]' \
  --policy.type=act \
  --policy.device=mps \
  --policy.chunk_size=100 \
  --policy.n_action_steps=20 \
  --policy.temporal_ensemble_coeff=0.01 \
  --steps=10000 \
  --batch_size=8 \
  --output_dir=outputs/act_insertion_ensemble
