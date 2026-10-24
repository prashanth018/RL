"""Step 1b: pull a 100-step action chunk, the thing ACT actually predicts.

Reads actions straight from the parquet (ds.hf_dataset) so this runs with no
video decoding. Images come later.
"""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# REPO_ID = "lerobot/aloha_sim_insertion_human"
REPO_ID = "lerobot/aloha_sim_transfer_cube_human"

CHUNK = 100  # ACT's chunk size

ds = LeRobotDataset(REPO_ID)
hf = ds.hf_dataset.with_format("torch")

actions = hf[:]["action"]  # every action in the dataset, concatenated
ep_idx = hf[:]["episode_index"]

print(f"all actions        : {tuple(actions.shape)}   (frames, dof)")

print(actions)

# episode 0 only, so the chunk never straddles an episode boundary
ep0 = actions[ep_idx == 0]
print(f"episode 0 actions  : {tuple(ep0.shape)}")

# the chunk ACT predicts at timestep t
t = 0
chunk = ep0[t : t + CHUNK]
print(f"chunk at t={t}      : {tuple(chunk.shape)}   <-- [chunk, dof]")

# what a batch of 8 such chunks looks like going into the decoder
batch = torch.stack([ep0[i : i + CHUNK] for i in range(8)])
print(f"batch of 8 chunks  : {tuple(batch.shape)}   <-- [batch, chunk, dof]")

# the overlap table from the survey notes, made concrete:
# action at absolute timestep 4 appears in the chunks predicted at t=0..4
print(f"\naction at t=4, as seen from each inference step:")
for infer_t in range(5):
    print(f"  predicted at t={infer_t} -> offset {4 - infer_t:>3} in its chunk")
print("  ^ 5 predictions of the same action = what temporal ensembling averages")
