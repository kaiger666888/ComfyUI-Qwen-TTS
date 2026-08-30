#!/usr/bin/env python3
"""
Demo: Generate TTS audio using ComfyUI-Qwen-TTS models directly.
Models are loaded from /data/models/comfyui/qwen-tts (local download target).
"""
import os
import sys
import torch
import soundfile as sf

# Ensure qwen_tts package is importable from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel

MODEL_ROOT = "/data/models/comfyui/qwen-tts"
CUSTOM_VOICE_MODEL = os.path.join(MODEL_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice")

def main():
    # Use CPU to avoid OOM on the 8GB GPU; fp32 for CPU compatibility
    device = "cpu"
    dtype = torch.float32
    print(f"Loading CustomVoice model from {CUSTOM_VOICE_MODEL} ...")
    print(f"Device: {device}, dtype: {dtype}")

    model = Qwen3TTSModel.from_pretrained(
        CUSTOM_VOICE_MODEL,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )

    # Show supported speakers
    speakers = model.model.get_supported_speakers()
    print("Supported speakers:", speakers)

    text = "你好，这是使用 ComfyUI-Qwen-TTS 项目生成的示例语音。声音自然流畅，适合各种中文语音合成场景。"
    speaker = "Ryan"  # preset speaker
    language = "chinese"

    print(f"\nGenerating speech ...")
    print(f"Text: {text}")
    print(f"Speaker: {speaker}")

    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        max_new_tokens=2048,
    )

    output_path = "/data/workspace/ComfyUI-Qwen-TTS/demo_output.wav"
    sf.write(output_path, wavs[0], sr)
    print(f"\nSaved audio to: {output_path}")
    print(f"Duration: {len(wavs[0]) / sr:.2f}s, Sample rate: {sr}Hz")

if __name__ == "__main__":
    main()
