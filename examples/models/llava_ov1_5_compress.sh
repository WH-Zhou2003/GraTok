export HF_HOME="~/.cache/huggingface"
#video_mmmu/videochatgpt/mvbench/nextqa_mc_test
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
export DECORD_EOF_RETRY_MAX=81920
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision1_5_compress \
    --model_args=pretrained=lmms-lab/LLaVA-OneVision-1.5-4B-Instruct,attn_implementation=flash_attention_2,max_pixels=3240000 \
    --tasks=videomme \
    --batch_size=1
