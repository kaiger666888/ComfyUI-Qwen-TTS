#!/usr/bin/env python3
"""
Demo: Generate a mature/yujie (御姐) style TTS voice using Qwen3-TTS VoiceDesign.
Models are loaded from /data/models/comfyui/qwen-tts.
"""
import os
import sys
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel

MODEL_ROOT = "/data/models/comfyui/qwen-tts"
VOICE_DESIGN_MODEL = os.path.join(MODEL_ROOT, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")

def main():
    device = "cpu"
    dtype = torch.float32
    print(f"Loading VoiceDesign model from {VOICE_DESIGN_MODEL} ...")
    print(f"Device: {device}, dtype: {dtype}")

    model = Qwen3TTSModel.from_pretrained(
        VOICE_DESIGN_MODEL,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )

    text = "晚上好，小家伙。今天过得怎么样？要不要陪我聊一会儿？"
    instruct = (
        "一位成熟优雅的御姐声音。音色低沉而富有磁性，语速舒缓，咬字清晰，"
        "尾音微微上扬，带着一丝慵懒和自信。语气温柔中透着一点强势，"
        "像一位阅历丰富的姐姐在耳边轻声说话。"
    )
    language = "chinese"

    print(f"\nGenerating speech ...")
    print(f"Text: {text}")
    print(f"Instruct: {instruct}")

    wavs, sr = model.generate_voice_design(
        text=text,
        instruct=instruct,
        language=language,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        max_new_tokens=2048,
    )

    output_path = "/data/workspace/ComfyUI-Qwen-TTS/demo_yujie_output.wav"
    sf.write(output_path, wavs[0], sr)
    print(f"\nSaved audio to: {output_path}")
    print(f"Duration: {len(wavs[0]) / sr:.2f}s, Sample rate: {sr}Hz")

if __name__ == "__main__":
    main()
