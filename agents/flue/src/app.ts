import { createAgentRouter } from "@flue/runtime/routing";
import { Hono } from "hono";
import { HfS2s } from "./agents/hf-s2s.ts";

const app = new Hono();

app.route("/agents/hf-s2s", createAgentRouter(HfS2s));

export default app;
