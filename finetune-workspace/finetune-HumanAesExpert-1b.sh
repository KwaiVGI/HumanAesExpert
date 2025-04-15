# set your gpu id, and NPROC_PER_NODE == the number of CUDA_VISIBLE_DEVICES
# if your gpu support flash_attention, you can set it to true
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 USE_HF=1 swift sft \
--custom_register_path ./HumanAesExpert_register.py \
--model_type internvl2_1b_HumanAesExpert \
--dataset ./HumanBeauty-trainset.jsonl \
--max_length 4096 \
--num_train_epochs 1 \
--sft_type lora \
--use_flash_attn false \
--additional_trainable_parameters language_model.lm_regression_head language_model.expert_head \
--deepspeed default-zero2 \
--dtype fp16 \
