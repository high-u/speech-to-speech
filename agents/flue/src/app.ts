import { createAgentRouter } from "@flue/runtime/routing";
import { Hono } from "hono";
import KuroshiroImport from "kuroshiro";
import KuromojiAnalyzerImport from "kuroshiro-analyzer-kuromoji";
import { HfS2s } from "./agents/hf-s2s.ts";

// Both packages are CJS with a Babel-style `exports.default`; under plain Node
// ESM the default import is the whole exports object, while bundler interop
// (vite/esbuild) already unwraps it. Handle both.
const Kuroshiro = (KuroshiroImport as unknown as { default?: typeof KuroshiroImport }).default ?? KuroshiroImport;
const KuromojiAnalyzer =
  (KuromojiAnalyzerImport as unknown as { default?: typeof KuromojiAnalyzerImport }).default ?? KuromojiAnalyzerImport;

const app = new Hono();

// "0" turns the conversion off; any other value, including an unset variable,
// leaves it on.
const hiraganaEnabled = process.env.HIRAGANA_CONVERSION !== "0";

const kuroshiroReady: Promise<InstanceType<typeof Kuroshiro>> | null = hiraganaEnabled
  ? (async () => {
      const kuroshiro = new Kuroshiro();
      await kuroshiro.init(new KuromojiAnalyzer());
      return kuroshiro;
    })()
  : null;
// Init starts at module load, long before the first request awaits it. Without a
// handler attached here, a failure there is an unhandled rejection that takes the
// process down; awaiting callers still see it and fall back to the raw text.
kuroshiroReady?.catch(() => {});


interface TextDeltaEvent {
  type: "message-delta";
  kind: "text";
  messageId: string;
  delta: string;
}

function isTextDelta(event: unknown): event is TextDeltaEvent {
  const e = event as Partial<TextDeltaEvent> | null;
  return !!e && e.type === "message-delta" && e.kind === "text" && typeof e.delta === "string";
}

async function toSpeechText(text: string): Promise<string> {
  try {
    const kuroshiro = await kuroshiroReady;
    if (!kuroshiro) return text;
    return await kuroshiro.convert(text, { to: "hiragana" });
  } catch (error) {
    console.error("[speech-text] conversion failed, sending the text unconverted:", error);
    return text;
  }
}

function createSpeechTextStream(): TransformStream<Uint8Array, Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let frameCarry = "";
  let textCarry = "";
  let carryMessageId: string | null = null;
  let rawLog = "";

  async function handleFrame(frame: string, controller: TransformStreamDefaultController<Uint8Array>) {
    const lines = frame.split("\n");
    const dataLineIndex = lines.findIndex((line) => line.startsWith("data:"));
    if (dataLineIndex === -1) {
      controller.enqueue(encoder.encode(`${frame}\n\n`));
      return;
    }

    // SSE allows one optional space after the colon. flue sends none; keep
    // whichever form arrived so the frame is rewritten exactly as it came.
    const dataLine = lines[dataLineIndex];
    const valueStart = dataLine.startsWith("data: ") ? 6 : 5;

    let payload: unknown;
    try {
      payload = JSON.parse(dataLine.slice(valueStart));
    } catch {
      controller.enqueue(encoder.encode(`${frame}\n\n`));
      return;
    }
    if (!Array.isArray(payload)) {
      controller.enqueue(encoder.encode(`${frame}\n\n`));
      return;
    }

    const outEvents: unknown[] = [];
    for (const event of payload) {
      if (isTextDelta(event)) {
        if (carryMessageId !== event.messageId) {
          textCarry = "";
          carryMessageId = event.messageId;
        }
        // Hold the whole message: the reading a morphological analyzer picks
        // depends on surrounding context, so it is converted once, in full.
        // Nothing downstream acts on partial text, so this costs no latency.
        rawLog += event.delta;
        textCarry += event.delta;
        continue;
      }
      if (textCarry && carryMessageId !== null) {
        outEvents.push({ type: "message-delta", kind: "text", messageId: carryMessageId, delta: await toSpeechText(textCarry) });
        textCarry = "";
      }
      outEvents.push(event);
    }

    if (outEvents.length > 0) {
      lines[dataLineIndex] = `${dataLine.slice(0, valueStart)}${JSON.stringify(outEvents)}`;
      controller.enqueue(encoder.encode(`${lines.join("\n")}\n\n`));
    }
  }

  return new TransformStream({
    async transform(chunk, controller) {
      frameCarry += decoder.decode(chunk, { stream: true });
      const frames = frameCarry.split("\n\n");
      frameCarry = frames.pop() ?? "";
      for (const frame of frames) await handleFrame(frame, controller);
    },
    async flush(controller) {
      frameCarry += decoder.decode();
      if (frameCarry) await handleFrame(frameCarry, controller);
      if (textCarry && carryMessageId !== null) {
        const payload = [{ type: "message-delta", kind: "text", messageId: carryMessageId, delta: await toSpeechText(textCarry) }];
        // The consumer only parses frames carrying an `event: data` line.
        controller.enqueue(encoder.encode(`event: data\ndata:${JSON.stringify(payload)}\n\n`));
      }
      console.log("[speech-text] before:", JSON.stringify(rawLog));
    },
  });
}

// With the conversion off, the stream is left alone entirely: flue's frames
// reach speech-to-speech exactly as they were sent.
if (hiraganaEnabled) {
  app.use("/agents/hf-s2s/*", async (c, next) => {
    await next();
    if (!c.res.body) return;
    if (!c.res.headers.get("content-type")?.startsWith("text/event-stream")) return;
    c.res = new Response(c.res.body.pipeThrough(createSpeechTextStream()), c.res);
  });
}

app.route("/agents/hf-s2s", createAgentRouter(HfS2s));

export default app;
