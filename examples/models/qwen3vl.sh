export HF_HOME="~/.cache/huggingface"
# export HF_HOME="/ssd/workspace/zwh/lmms-eval-main/datasets/NExTQA"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)
export DECORD_EOF_RETRY_MAX=81920
# Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 
    # --tasks "mmmu_val,mmbench_en_dev,ocrbench,realworldqa,mmstar" \flash_attention_2
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=Qwen/Qwen3-VL-4B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks "mvbench" \
    --batch_size 1
