export USE_TF=0
lerobot-eval \
  --policy.path=outputs/act_transfer_cube_generic/checkpoints/001000/pretrained_model \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=4 \
  --eval.batch_size=1 \
  --policy.device=mps
