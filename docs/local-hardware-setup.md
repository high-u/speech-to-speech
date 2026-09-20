# ローカル環境のセットアップメモ

## llama.cpp で動かす 

```bash
cd speech-to-speech
uv sync
```

```bash
CUDA_VISIBLE_DEVICES=0 \
llama-server -hf google/gemma-4-E4B-it-qat-q4_0-gguf \
  -c 32768 -ngl 99 -fa on -cb -np 1 --temp 1.0 --top-p 0.95 --top-k 64
```

```bash
cd speech-to-speech

PULSE_SOURCE=alsa_input.usb-USB_Microphone_Maono_Fairy_2018_06_26-00.iec958-stereo \
PULSE_SINK=alsa_output.pci-0000_70_00.1.hdmi-stereo \
CUDA_VISIBLE_DEVICES=1 uv run speech-to-speech local \
  --stt whisper \
  --stt_model_name openai/whisper-large-v3 \
  --stt_device cuda \
  --stt_torch_dtype float16 \
  --language ja \
  --llm_backend chat-completions \
  --model_name local-jp \
  --responses_api_base_url http://127.0.0.1:8080/v1 \
  --responses_api_api_key "" \
  --responses_api_stream \
  --tts qwen3 \
  --qwen3_tts_backend torch \
  --qwen3_tts_device cuda \
  --qwen3_tts_language ja \
  --init_chat_prompt "あなたは日本語で話す音声アシスタントです。返答は簡潔に、1〜2文で。" \
  --enable_live_transcription \
  --local_audio_input_device 18 \
  --local_audio_output_device 18
```

## マイクとスピーカーの指定

```bash
uv run python -m sounddevice
```
出力の中から名前が`pulse`の行を探し、その番号を`--local_audio_input_device`/`--local_audio_output_device`に使う（例: `18`）。

```bash
pactl list short sources
```
出力例:
```
54   alsa_input.usb-Generic_USB_Audio-00.iec958-stereo
187  alsa_input.usb-USB_Microphone_Maono_Fairy_2018_06_26-00.iec958-stereo
```
`alsa_input.`で始まる行の中から、使いたいマイクの行を選び、2列目をそのまま`PULSE_SOURCE`に使う。

```bash
pactl list short sinks
```
出力例:
```
53   alsa_output.usb-Generic_USB_Audio-00.iec958-stereo
55   alsa_output.pci-0000_70_00.1.hdmi-stereo
186  alsa_output.usb-USB_Microphone_Maono_Fairy_2018_06_26-00.iec958-stereo
```
`alsa_output.`で始まる行の中から、使いたいスピーカーの行を選び、2列目をそのまま`PULSE_SINK`に使う。

