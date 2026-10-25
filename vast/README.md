# Vast

```bash
# Mac: copy files up
scp -P 19637 vast/vast_setup.sh act/act_aloha_train.sh root@137.175.22.196:~/

# Mac: log in
ssh -p 19637 root@137.175.22.196

# box
tmux ls          # see the session name
tmux attach      # attaches to the most recent one, no name needed
bash vast_setup.sh
wandb login
bash act_aloha_train.sh        # Ctrl-b d to detach

# Mac: copy checkpoint home
rsync -avz -e "ssh -p <PORT>" root@<HOST>:~/outputs/<run>/ ./outputs/<run>/
```

Then **destroy** the instance.
