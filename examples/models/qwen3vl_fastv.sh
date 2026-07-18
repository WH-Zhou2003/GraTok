export HF_HOME="~/.cache/huggingface"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)

# Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 
#video_mmmu/videochatgpt/mvbench/nextqa_mc_test/
export DECORD_EOF_RETRY_MAX=81920
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl_compress\
    --model_args=pretrained=Qwen/Qwen3-VL-2B-Instruct,max_pixels=12845056,attn_implementation=eager,interleave_visuals=False \
    --tasks "nextqa_mc_test" \
    --batch_size 1 \
    --output_path ./logs/