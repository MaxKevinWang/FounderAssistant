import axios from "axios";

export type RAGDecisionDomain =
  | "legal_form_selection"
  | "funding_preparation"
  | "market_validation";

export interface RAGConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export type AdvisorRequestAction = "answer" | "answer_with_warning" | "clarify" | "refuse_and_redirect";

export interface AdvisorRequestAssessment {
  action: AdvisorRequestAction;
  language: "en" | "de" | "zh" | "other";
  risk_level: "low" | "medium" | "high";
  risk_categories: string[];
  questionable_shortcut: boolean;
  sensitive_categories: string[];
  needs_clarification: boolean;
  clarification_question: string;
  source: "llm_structured" | "deterministic_rule" | "fallback_rule";
  guard_model: string;
  guard_latency_ms: number;
  guard_usage: Record<string, number>;
  message: string;
  notice: string;
  generation_instruction: string;
}

// Typed contract for local generator status endpoint.
export interface LocalLLMStatus {
  reachable: boolean;
  base_url?: string;
  configured_model?: string;
  installed?: boolean;
  available_models?: string[];
  generation_models?: string[];
  embedding_models?: string[];
  error?: string;
}

export interface LocalLLMChatResponse {
  model: string;
  response: string;
  done: boolean;
  latency_ms: number;
}

export interface RAGCitationItem {
  rank: number;
  score: number;
  domain: string;
  source_path: string;
  source_name: string;
  chunk_index: number;
  text_preview: string;
  chunk_type?: string;
  topic?: string;
  legal_area?: string;
  business_type?: string;
  authority?: string;
  source_quality?: string;
  language?: string;
  jurisdiction?: string;
  document_type?: string;
  section_title?: string;
  heading_path?: string;
  page_start?: string;
  page_end?: string;
  url?: string;
  publication_date?: string;
  law_name?: string;
  law_abbreviation?: string;
  law_section?: string;
  effective_from?: string;
  effective_to?: string;
  retrieved_at?: string;
  imported_at?: string;
  canonical_url?: string;
  temporal_status?: string;
  supersedes?: string;
  metadata_reviewed?: boolean;
}

export interface RAGRuntimeProfile {
  model_type: "dense" | "expert";
  top_k: number;
  max_tokens: number;
  auto_applied: boolean;
  domain_filter_applied?: boolean;
  domain?: RAGDecisionDomain;
}

export interface RAGStreamMetadata {
  used_rag: boolean;
  retrieval_latency_ms: number;
  retrieval_mode?: string;
  retrieval_domain?: RAGDecisionDomain | "";
  history_message_count?: number;
  estimated_history_tokens?: number;
  history_used_for_retrieval?: boolean;
  history_token_estimator?: string;
  citations: RAGCitationItem[];
}

export interface RAGStreamDone {
  provider?: string;
  model: string;
  done: boolean;
  finish_reason?: string;
  latency_ms: number;
}

export interface RAGStreamHandlers {
  onProfile?: (profile: RAGRuntimeProfile) => void;
  onMetadata?: (metadata: RAGStreamMetadata) => void;
  onToken?: (token: string) => void;
  onDone?: (done: RAGStreamDone) => void;
  onError?: (message: string) => void;
}

// Shared Axios instance for advisor-related API calls.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  // Institute-hosted models can cold-start or queue on shared GPUs. Keep the
  // browser request alive long enough for slow but valid Open WebUI responses.
  // The backend generation timeout is 360 seconds. Leave an additional
  // transport margin so the browser can receive a backend timeout or fallback.
  timeout: 390000,
});

export async function fetchLocalLLMStatus(): Promise<LocalLLMStatus> {
  // Baseline local-model health and installation metadata.
  const { data } = await api.get<LocalLLMStatus>("/api/advisors/local-llm/status/");
  return data;
}

export async function chatWithLocalLLM(payload: {
  prompt: string;
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
  model?: string;
}): Promise<LocalLLMChatResponse> {
  // Baseline generation endpoint (no retrieval augmentation).
  const { data } = await api.post<LocalLLMChatResponse>("/api/advisors/local-llm/chat/", payload);
  return data;
}

export async function assessAdvisorRequest(payload: {
  prompt: string;
  domain?: RAGDecisionDomain;
  history?: RAGConversationMessage[];
  model?: string;
}): Promise<AdvisorRequestAssessment> {
  const { data } = await api.post<AdvisorRequestAssessment>("/api/advisors/request-assessment/", payload);
  return data;
}

function parseSSEFrame(rawFrame: string): { event: string; data: string } | null {
  const lines = rawFrame.split(/\r?\n/);
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (!dataLines.length) return null;
  return { event, data: dataLines.join("\n") };
}

async function dispatchSSEFrame(rawFrame: string, handlers: RAGStreamHandlers): Promise<void> {
  const frame = parseSSEFrame(rawFrame);
  if (!frame) return;

  const payload = JSON.parse(frame.data) as Record<string, unknown>;
  if (frame.event === "profile") {
    handlers.onProfile?.(payload as unknown as RAGRuntimeProfile);
    return;
  }
  if (frame.event === "metadata") {
    handlers.onMetadata?.(payload as unknown as RAGStreamMetadata);
    return;
  }
  if (frame.event === "token") {
    const token = typeof payload.content === "string" ? payload.content : "";
    if (token) handlers.onToken?.(token);
    return;
  }
  if (frame.event === "done") {
    handlers.onDone?.(payload as unknown as RAGStreamDone);
    return;
  }
  if (frame.event === "error") {
    const message = typeof payload.error === "string" ? payload.error : "Streaming request failed.";
    handlers.onError?.(message);
    throw new Error(message);
  }
}

export async function streamChatWithRAG(
  payload: {
    prompt: string;
    system_prompt?: string;
    temperature?: number;
    max_tokens?: number;
    top_k?: 5;
    domain?: RAGDecisionDomain;
    history?: RAGConversationMessage[];
    model?: string;
  },
  handlers: RAGStreamHandlers,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/advisors/rag/chat/stream/`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Streaming request failed with HTTP ${response.status}.`);
  }

  if (!response.body) {
    throw new Error("Streaming response did not include a readable body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawFrame = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (rawFrame) {
        await dispatchSSEFrame(rawFrame, handlers);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    await dispatchSSEFrame(buffer.trim(), handlers);
  }
}
