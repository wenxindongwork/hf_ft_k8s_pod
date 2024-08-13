rm -rf profile/
tmux capture-pane -pS - > ~/tmux-buffer.txt 
cp ~/tmux-buffer.txt tmux-buffer.txt 
cp /finetune_long_prof.py finetune_long_prof.py
cp /usr/local/lib/python3.10/site-packages/transformers/trainer.py trainer.py
cp /llama8b_sft_trainer.py llama8b_sft_trainer.py
cp /gemma.py gemma.py
cp /llama8b_trainer.py llama8b_trainer.py
cp -r /tmp/profile/plugins/profile profile
cp -r /tmp/trace* trace/

# cp /tmp/save1.ir.0 save1.ir.0
