"""Step 1: just look at the data. No model, no training.

Goal: see what one sample of an ALOHA demo actually contains, and then
pull a 100-step action chunk out of it (the thing ACT predicts).
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "lerobot/aloha_sim_insertion_human"

# --- 1. plain dataset: one sample = one timestep ---
ds = LeRobotDataset(REPO_ID, video_backend="pyav")

print(f"episodes : {ds.num_episodes}")
print(f"frames   : {ds.num_frames}")
print(f"fps      : {ds.fps}")
print("\nfeatures:")
for name, spec in ds.features.items():
    print(f"  {name:<32} {spec.get('shape')}  {spec.get('dtype')}")

print("\none sample (ds[0]):")
for k, v in ds[0].items():
    shape = getattr(v, "shape", None)
    print(f"  {k:<32} {tuple(shape) if shape is not None else v}")
