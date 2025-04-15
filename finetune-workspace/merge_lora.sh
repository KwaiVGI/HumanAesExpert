CUDA_VISIBLE_DEVICES=0 USE_HF=1 swift export \
    --custom_register_path ./HumanAesExpert_register.py \
    --model_type internvl2_8b_HumanAesExpert \
    --ckpt_dir "./output/internvl2_8b_HumanAesExpert/v3-20250302-151103/checkpoint-12006" \
    --merge_lora true