# flue バックエンド

## 起動

### llama-server

```bash
CUDA_VISIBLE_DEVICES=0 \
llama-server -hf unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL \
  -c 16384 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -fa on -ctk q8_0 -ctv q8_0 -np 1 --fit off \
  --n-cpu-moe 99 \
  --jinja --chat-template-kwargs '{"enable_thinking": false}' \
  --alias "assistant-model"
```

`--alias`の値が、次で登録するモデルプロバイダの`model_id`になる。

### flue

モデルプロバイダやagentは、API/UIから後付け登録するのではなく、**プロジェクトのTypeScriptコードとして書く**。実装は`agents/flue/`にある（`flue init`でスキャフォールドし、agentは`src/agents/hf-s2s.ts`、ルーティングは`src/app.ts`）。

llama-serverのbaseUrl/contextWindow/maxTokensは`agents/flue/.env`の`LLAMA_SERVER_BASE_URL`/`LLAMA_SERVER_CONTEXT_WINDOW`/`LLAMA_SERVER_MAX_TOKENS`で設定する。

**このアダプターは`/agents/{agent_name}/{conversationId}`という形のパスに固定で叩きに行く**（`flue_language_model.py`の`_stream_turn`にハードコード）。`createAgentRouter`自体はどこにマウントしてもよい設計だが、`src/app.ts`でのマウント先（`/agents/hf-s2s`）を変えると、`--flue_base_url`/`--flue_agent_name`をどう組み合わせても届かない。

初回のみ、依存をインストールし、`.env`を作る。

```bash
cd agents/flue
npm install
cp .env.example .env
```

起動する。

```bash
cd agents/flue
npm run dev
```

<http://localhost:5173>でHTTP APIが立つ。管理UIは無い。

**agentを後から変えるときは、コードを直して起動し直すだけ。** ローカルの`vite dev`では`Authorization`ヘッダ無しで叩ける（実機で確認済み）。

**ツール承認待ちの機構は無い。** `@flue/runtime`が公開する全16個のフック（`useTool`/`useSandbox`/`useSkill`/`useSubagent`/`useMcpConnection`/`useModel`/`useAgentStart`/`useAgentFinish`/`useResponseStart`/`useResponseFinish`/`useDelivery`/`useDispatchMessage`/`useDataWriter`/`useInitialData`/`useInstruction`/`usePersistentState`）を直接確認したが、承認・確認・保留に類する仕組みは1つも存在しない。

### speech-to-speech

初回のみ、依存をインストールする。

```bash
cd speech-to-speech
uv sync
```

```bash
cd speech-to-speech

CUDA_VISIBLE_DEVICES=0 \
PULSE_SOURCE=alsa_input.usb-USB_Microphone_Maono_Fairy_2018_06_26-00.iec958-stereo \
PULSE_SINK=alsa_output.pci-0000_70_00.1.hdmi-stereo \
uv run speech-to-speech local \
  --language ja \
  --stt whisper \
  --stt_model_name openai/whisper-large-v3 \
  --stt_device cuda \
  --stt_torch_dtype float16 \
  --llm_backend flue \
  --flue_base_url http://127.0.0.1:5173 \
  --flue_agent_name hf-s2s \
  --tts qwen3 \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_device cuda \
  --qwen3_tts_backend torch \
  --qwen3_tts_speaker Ono_anna \
  --local_audio_input_device 18 \
  --local_audio_output_device 18 \
  --local_audio_block_mic_during_playback true
```

（`PULSE_SOURCE`/`PULSE_SINK`やデバイス番号は機材固有の値。自分の環境での調べ方は[local-hardware-setup.md](./local-hardware-setup.md)参照）

`local`はサーバーとマイク・スピーカーのクライアントを1プロセスで動かすモード。停止はCtrl-C。

### 自分の声を拾わないために

`--local_audio_block_mic_during_playback true`
スピーカーとマイクで使う場合、これが無いと**エージェントが自分の声を拾って会話が暴走する**。既定は`false`。

`true`にすると、LLMの音声再生中だけマイク入力が無効化され、再生が終わると有効に戻る。この間は割り込み（barge-in）ができない。

イヤホンで聞く場合はマイクにLLMの声が回り込まないため、このオプションを付けなくても（`false`のままでも）暴走せず、割り込みも可能なまま使える。

## 注意事項

speech-to-speech側の実装に起因して、次のことが起きる。

### 発話できない文字は削除される

speech-to-speech側は、TTSに渡す前に発話可能な文字のリストでテキストを濾し、リストに無い文字を削除する（`src/speech_to_speech/LLM/utils.py`の`SPEECHABLE_PATTERN`）。全角チルダ`〜`や`℃`はリストに含まれていないため、消える。

```
80%〜90% → 80%90%
22℃      → 22
```

天気予報のように単位や範囲を含む回答では、意味が変わって聞こえる。

### 漢字はひらがなに変換してから渡している

TTSに漢字を含むテキストを渡すと、日本語ではない読み方になる。これを防ぐため、flue側でSSEのテキストをkuromoji（形態素解析）とkuroshiro（かな変換）でひらがなに変換してからspeech-to-speechへ送っている（`agents/flue/src/app.ts`）。

### 音声は全文の生成が終わってから始まる

speech-to-speech側は、受け取ったテキストをnltkの`sent_tokenize`で文に区切り、確定した文が`--stream_batch_sentences`（既定3）個たまった時点でTTSに渡す。この`sent_tokenize`は英語用で、`.` `!` `?`しか文末として扱わず、「。」では区切らない。

日本語の回答は最後まで区切られないため、この送出は一度も発生せず、ターン終了時に全文が一括でTTSへ渡る。`--stream_batch_sentences`を変えても効果はない。
