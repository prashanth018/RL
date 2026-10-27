export USE_TF=0
lerobot-eval \
  --policy.path=outputs/act_transfer_cube_chunk40_te_80k \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --policy.device=mps \
  --output_dir=outputs/eval/chunk40_te_80k
