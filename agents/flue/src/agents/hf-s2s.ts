"use agent";
import { createProvider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { setProvider, useModel, useSandbox } from "@flue/runtime";
import { local } from "@flue/runtime/node";

setProvider(
  createProvider({
    id: "llama-server",
    auth: {
      apiKey: {
        name: "llama.cpp keyless",
        resolve: async () => ({ auth: { apiKey: "unused" } }),
      },
    },
    models: [
      {
        id: "assistant-model",
        name: "assistant-model",
        api: "openai-completions",
        provider: "llama-server",
        baseUrl: process.env.LLAMA_SERVER_BASE_URL as string,
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: Number(process.env.LLAMA_SERVER_CONTEXT_WINDOW),
        maxTokens: Number(process.env.LLAMA_SERVER_MAX_TOKENS),
      },
    ],
    api: openAICompletionsApi(),
  }),
);

export function HfS2s() {
  useModel("llama-server/assistant-model");
  useSandbox(local());
}
