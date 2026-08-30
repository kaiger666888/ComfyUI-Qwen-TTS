#!/usr/bin/env python3
"""
Demo: Clone the yujie voice from the previously generated audio,
then synthesize new (non-explicit) content with that voice.
"""
import os
import sys
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel

MODEL_ROOT = "/data/models/comfyui/qwen-tts"
BASE_MODEL = os.path.join(MODEL_ROOT, "Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO = "/data/workspace/ComfyUI-Qwen-TTS/demo_yujie_output.wav"

def main():
    device = "cpu"
    dtype = torch.float32
    print(f"Loading Base model from {BASE_MODEL} ...")
    print(f"Device: {device}, dtype: {dtype}")

    model = Qwen3TTSModel.from_pretrained(
        BASE_MODEL,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )

    ref_text = "晚上好，小家伙。今天过得怎么样？要不要陪我聊一会儿？"
    target_text = "能不能过来陪我一下，我现在好想你，想跟你说说话。"
    language = "chinese"

    print(f"\nCloning voice from reference audio ...")
    print(f"Ref audio: {REF_AUDIO}")
    print(f"Target text: {target_text}")

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language=language,
        ref_audio=REF_AUDIO,
        ref_text=ref_text,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        max_new_tokens=2048,
    )

    output_path = "/data/workspace/ComfyUI-Qwen-TTS/demo_cloned_yujie_output.wav"
    sf.write(output_path, wavs[0], sr)
    print(f"\nSaved cloned audio to: {output_path}")
    print(f"Duration: {len(wavs[0]) / sr:.2f}s, Sample rate: {sr}Hz")

if __name__ == "__main__":
    main()
