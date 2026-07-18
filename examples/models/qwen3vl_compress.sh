export HF_HOME="/data_sdc/data/zwh/dataset"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)
export CUDA_VISIBLE_DEVICES=1
# Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 
#video_mmmu/videochatgpt/mvbench/nextqa_mc_test/PLM-VideoBench/videott/videomme
# export DECORD_EOF_RETRY_MAX=81920
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl_compress\
    --model_args=pretrained=Qwen/Qwen3-VL-2B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks "nextqa" \
    --batch_size 1 \
    --output_path ./logs/