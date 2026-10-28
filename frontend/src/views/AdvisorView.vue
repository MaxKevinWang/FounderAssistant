<template>
  <section class="consumer-view">
    <div class="consumer-chat-surface" :class="{ 'consumer-chat-surface-empty': conversation.length === 0 }">
      <div ref="threadRef" class="chat-thread" :class="{ 'chat-thread-empty': conversation.length === 0 }">
        <div v-if="conversation.length === 0" class="consumer-empty-identity">
          <span class="consumer-empty-logo" aria-hidden="true">
            <svg class="composer-icon" viewBox="0 0 24 24">
              <path d="M5 7h14v10H9l-4 3V7Z" />
              <path d="M9 11h6M9 14h3" />
            </svg>
          </span>
          <h2>{{ activeConsumerModelDisplay }}</h2>
        </div>

        <article
          v-for="message in conversation"
          :key="message.id"
          class="chat-message"
          :class="`chat-message-${message.role}`"
        >
          <div class="chat-avatar">{{ message.role === "user" ? "You" : "AI" }}</div>
          <div class="chat-bubble" :class="{ 'chat-bubble-error': message.error }">
            <div
              v-if="message.role === 'assistant'"
              class="markdown-body"
              v-html="renderMarkdown(message.content)"
            ></div>
            <div v-if="message.role === 'assistant' && message.streaming" class="streaming-cursor" aria-live="polite">
              <span></span>
            </div>
            <pre v-if="message.role === 'user'" class="response-block">{{ message.content }}</pre>

            <div v-if="message.role === 'assistant' && message.requestAssessment" class="guard-status-row">
              <span v-if="message.requestAssessment.action === 'clarify'" class="guard-badge">Clarification requested</span>
              <span v-if="message.requestAssessment.risk_level === 'high'" class="guard-badge guard-badge-warning">High-risk guidance</span>
              <span v-if="message.requestAssessment.questionable_shortcut" class="guard-badge guard-badge-blocked">Unsafe shortcut declined</span>
              <span v-if="message.requestAssessment.sensitive_categories.length" class="guard-badge guard-badge-sensitive">Sensitive information detected</span>
              <span
                v-if="message.requestAssessment.source === 'fallback_rule'"
                class="guard-badge guard-badge-warning"
                title="The Guard model did not return a valid assessment. Deterministic safety rules were used instead."
              >Basic safety check used</span>
            </div>

            <div
              v-if="message.role === 'assistant' && message.content.trim()"
              class="message-action-row"
              aria-label="Assistant response actions"
            >
              <button
                class="message-action-btn"
                type="button"
                :title="copiedMessageId === message.id ? 'Copied' : 'Copy response'"
                :aria-label="copiedMessageId === message.id ? 'Copied response' : 'Copy response'"
                @click="copyAssistantMessage(message.id, message.content)"
              >
                <svg class="message-action-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8 8h10v12H8z" />
                  <path d="M6 16H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
                </svg>
                <span v-if="copiedMessageId === message.id" class="message-action-label">Copied</span>
              </button>
              <button
                class="message-action-btn"
                :class="{ active: message.feedback === 'up' }"
                type="button"
                title="Good response"
                aria-label="Mark response as good"
                @click="setAssistantFeedback(message.id, 'up')"
              >
                <svg class="message-action-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M7 10v10H4V10h3Z" />
                  <path d="M7 10l5-7 1.2.8a2 2 0 0 1 .8 1.8L13.5 9H19a2 2 0 0 1 2 2.3l-1 6A2 2 0 0 1 18 19H7" />
                </svg>
              </button>
              <button
                class="message-action-btn"
                :class="{ active: message.feedback === 'down' }"
                type="button"
                title="Poor response"
                aria-label="Mark response as poor"
                @click="setAssistantFeedback(message.id, 'down')"
              >
                <svg class="message-action-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M7 14V4H4v10h3Z" />
                  <path d="M7 14l5 7 1.2-.8a2 2 0 0 0 .8-1.8L13.5 15H19a2 2 0 0 0 2-2.3l-1-6A2 2 0 0 0 18 5H7" />
                </svg>
              </button>
            </div>

            <div class="chat-meta-row" v-if="message.role === 'assistant'">
              <p v-if="message.model || message.streaming" class="chat-meta">
                <template v-if="message.model">{{ message.model }} · </template>
                <span v-if="message.streaming">Generating {{ formatLatencySeconds(generationElapsedMs) }}</span>
                <span v-else>{{ formatLatencySeconds(message.latencyMs) }}</span>
              </p>
              <p class="chat-time">{{ formatMessageTime(message.createdAt) }}</p>
            </div>
            <p v-else class="chat-time">{{ formatMessageTime(message.createdAt) }}</p>

            <div v-if="message.role === 'assistant' && message.citations.length" class="citation-list">
              <p class="citation-title">Sources</p>
              <ul>
                <li v-for="item in message.citations" :key="`${message.id}-${item.rank}-${item.chunk_index}`">
                  <a
                    v-if="item.url || item.canonical_url"
                    :href="item.canonical_url || item.url"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {{ formatCitationSourceName(item.source_name || item.source_path) }}
                  </a>
                  <span v-else>{{ formatCitationSourceName(item.source_name || item.source_path) }}</span>
                  <span v-if="displayPublicationDate(item)" class="citation-date">
                    · Source date {{ formatSourceDate(displayPublicationDate(item)) }}
                  </span>
                  <span v-if="displayCollectionDate(item)" class="citation-date">
                    · Collected {{ formatSourceDate(displayCollectionDate(item)) }}
                  </span>
                  <span v-else-if="item.imported_at && !displayPublicationDate(item)" class="citation-date">
                    · Imported {{ formatSourceDate(item.imported_at) }}
                  </span>
                </li>
              </ul>
            </div>
          </div>
        </article>

        <article
          v-if="pendingDomainConfirmation?.sessionId === chatStore.activeSessionId"
          class="chat-message chat-message-assistant"
        >
          <div class="chat-avatar">AI</div>
          <div class="chat-bubble domain-confirmation-card">
            <p class="domain-confirmation-title">Choose a topic or let me search all topics</p>
            <p class="domain-confirmation-copy">If you are not sure, I can search the complete knowledge base.</p>
            <div class="domain-choice-grid">
              <button
                v-for="item in decisionDomains"
                :key="`confirm-domain-${item.value}`"
                class="domain-choice-btn"
                type="button"
                :disabled="isConsumerBusy"
                @click="confirmPendingDomain(item.value)"
              >
                {{ item.label }}
              </button>
              <button
                class="domain-choice-btn"
                type="button"
                :disabled="isConsumerBusy"
                @click="confirmPendingDomain()"
              >
                Not sure · Search all topics
              </button>
            </div>
          </div>
        </article>

        <article v-if="shouldShowTypingBubble" class="chat-message chat-message-assistant">
          <div class="chat-avatar">AI</div>
          <div class="chat-bubble typing-bubble">
            <div class="typing-dots" aria-hidden="true">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <p class="typing-status">Generating... {{ formatLatencySeconds(generationElapsedMs) }}</p>
          </div>
        </article>
      </div>
      <form
        class="chat-input-dock"
        :class="{ 'chat-input-dock-empty': conversation.length === 0 }"
        @submit.prevent="submitConsumerForm"
      >
        <textarea
          ref="consumerInputRef"
          v-model="consumerInput"
          rows="1"
          class="chat-textarea"
          aria-label="Message"
          placeholder="How can I help with your startup?"
          @input="resizeConsumerInput"
          @keydown="onConsumerInputKeydown"
        />
        <p v-if="consumerError" class="error-text composer-error">{{ consumerError }}</p>
        <div class="chat-actions">
          <label class="composer-menu composer-topic" title="Choose a topic">
            <svg class="composer-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="m12 3 3 3-3 3-3-3 3-3Zm-6 6 3 3-3 3-3-3 3-3Zm12 0 3 3-3 3-3-3 3-3Zm-6 6 3 3-3 3-3-3 3-3Z" />
            </svg>
            <span class="composer-menu-text">{{ activeConsumerTopicDisplay }}</span>
            <svg class="composer-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4" /></svg>
            <select
              id="consumerDomain"
              v-model="consumerDomain"
              class="composer-select"
              aria-label="Retrieval topic"
            >
                <option value="ask">Choose topic after sending</option>
                <option value="auto">Not sure · Search all topics</option>
              <option v-for="item in decisionDomains" :key="item.value" :value="item.value">
                {{ item.label }}
              </option>
            </select>
          </label>
          <label class="composer-menu composer-model" title="Choose a model">
            <svg class="composer-icon" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="6" y="6" width="12" height="12" rx="3" />
              <path d="M10 3v3m4-3v3m-4 12v3m4-3v3M3 10h3m-3 4h3m12-4h3m-3 4h3" />
            </svg>
            <span class="composer-menu-text">{{ activeConsumerModelDisplay }}</span>
            <svg class="composer-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4" /></svg>
            <select
              id="consumerModel"
              v-model="consumerModel"
              class="composer-select"
              aria-label="Generation model"
              :disabled="!availableModels.length"
            >
              <option value="">Auto model</option>
              <option v-for="modelName in availableModels" :key="modelName" :value="modelName">
                {{ modelName }}
              </option>
            </select>
          </label>

          <button
            class="chat-submit"
            :disabled="isConsumerBusy || hasPendingDomainConfirmation || hasPendingSensitiveReview || !consumerInput.trim()"
            :aria-label="isConsumerLoading ? 'Generating response' : 'Send message'"
            :title="isConsumerLoading ? 'Generating response' : 'Send message'"
            type="submit"
          >
            <svg class="composer-icon" viewBox="0 0 24 24" aria-hidden="true">
              <template v-if="isConsumerLoading"><circle cx="5" cy="12" r="1" /><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /></template>
              <path v-else d="M12 19V5m-6 6 6-6 6 6" />
            </svg>
          </button>
        </div>
      </form>

      <div v-if="conversation.length === 0" class="consumer-suggestions">
        <p class="suggestion-heading">
          <svg class="composer-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m13 2-9 12h7l-1 8 10-13h-7l1-7Z" /></svg>
          Suggestions
        </p>
        <div class="suggestion-grid">
          <button
            v-for="item in starterPrompts"
            :key="item.text"
            class="suggestion-btn"
            type="button"
            :disabled="isConsumerBusy"
            @click="sendStarterPrompt(item)"
          >
            <span class="suggestion-title">{{ item.title }}</span>
            <span class="suggestion-detail">{{ item.detail }}</span>
          </button>
        </div>
      </div>
    </div>

    <div v-if="pendingSensitiveReview" class="sensitive-review-overlay" role="presentation">
      <section class="sensitive-review-dialog" role="dialog" aria-modal="true" aria-labelledby="sensitive-review-title">
        <div class="sensitive-review-icon" aria-hidden="true">!</div>
        <h3 id="sensitive-review-title">Check sensitive information before sending</h3>
        <p>This message may contain {{ sensitiveFindingLabels }}. Only share information needed for your question.</p>
        <p class="sensitive-review-note">Detection happens in this browser. The identified values are not copied into the warning.</p>
        <div class="sensitive-review-actions">
          <button class="btn sensitive-edit-btn" type="button" @click="editSensitiveInput">Edit message</button>
          <button class="btn sensitive-continue-btn" type="button" @click="continueWithSensitiveInput">Send anyway</button>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import { type ConsumerChatFeedback, useConsumerChatStore } from "@/stores/consumerChat";
import { renderMarkdown } from "@/utils/markdown";
import { detectStructuredSensitiveInput, type SensitiveInputFinding } from "@/utils/sensitiveInput";
import {
  assessAdvisorRequest,
  chatWithLocalLLM,
  fetchLocalLLMStatus,
  streamChatWithRAG,
  type AdvisorRequestAssessment,
  type LocalLLMStatus,
  type RAGCitationItem,
  type RAGConversationMessage,
  type RAGDecisionDomain,
  type RAGRuntimeProfile,
} from "@/services/localLlm";

interface StarterPrompt {
  title: string;
  detail: string;
  text: string;
  domain: RAGDecisionDomain;
}

const chatStore = useConsumerChatStore();

const status = ref<LocalLLMStatus | null>(null);

// Temporary consumer-only override while testing truncated long-form answers.
// Evaluation runners and backend defaults remain fixed at 900 tokens.
const TEMPORARY_CONSUMER_MAX_TOKENS = 1400;

const systemPrompt = ref(
  "You are a founder-assistance system for early-stage entrepreneurs in Germany. Give detailed, practical, source-grounded guidance in clean Markdown. Answer strictly in the same language as the user's question and do not mix German and English unless the user does so. If the question is German, do not add English translations in parentheses. If the question is English, keep German legal and administrative terms such as Gewerbeamt, Finanzamt, IHK, GmbH, UG, and GbR unchanged and explain them briefly only when needed. Do not force a step-by-step format; use numbered actions only when a process is actually helpful. End with a short conclusion in the answer language.",
);

const threadRef = ref<HTMLElement | null>(null);
const consumerInputRef = ref<HTMLTextAreaElement | null>(null);
const consumerInput = ref("");
const consumerError = ref("");
const loadingSessionId = ref<string | null>(null);
const generationStartedAt = ref<number | null>(null);
const generationElapsedMs = ref(0);
const streamingMessageId = ref<number | null>(null);
const copiedMessageId = ref<number | null>(null);
const CONSUMER_MODEL_STORAGE_KEY = "founder-assistant-consumer-model";
const consumerModel = ref(readStoredConsumerModel());
const consumerDomain = computed<RAGDecisionDomain | "ask" | "auto">({
  get: () => chatStore.activeSession?.domain ?? "ask",
  set: (value) => {
    chatStore.setSessionDomain(
      chatStore.activeSessionId,
      value === "ask" ? undefined : value,
    );
  },
});
const pendingDomainConfirmation = ref<{
  sessionId: string;
  prompt: string;
  history: RAGConversationMessage[];
} | null>(null);
const pendingSensitiveReview = ref<{
  text: string;
  findings: SensitiveInputFinding[];
} | null>(null);
let generationTimer: number | undefined;
let copiedMessageTimer: number | undefined;

const MAX_HISTORY_TURNS = 3;
const MAX_HISTORY_TOKENS = 1500;
const MAX_HISTORY_MESSAGE_TOKENS = 600;

const decisionDomains: { value: RAGDecisionDomain; label: string }[] = [
  { value: "legal_form_selection", label: "Legal form and startup registration" },
  { value: "funding_preparation", label: "Funding preparation" },
  { value: "market_validation", label: "Market validation" },
];

const activeConsumerTopicDisplay = computed(() => {
  const labels: Record<RAGDecisionDomain, string> = {
    legal_form_selection: "Legal form",
    funding_preparation: "Funding",
    market_validation: "Market validation",
  };
  if (consumerDomain.value === "ask") return "Topic";
  if (consumerDomain.value === "auto") return "All topics";
  return labels[consumerDomain.value];
});

const starterPrompts: StarterPrompt[] = [
  {
    title: "Choose a legal form",
    detail: "Compare UG and GmbH with EUR 8,000",
    text: "I am in Germany with EUR 8,000. Should I start with UG or GmbH?",
    domain: "legal_form_selection",
  },
  {
    title: "Validate your market",
    detail: "Build an 8-week B2C validation plan",
    text: "Create an 8-week market validation plan for a B2C startup.",
    domain: "market_validation",
  },
  {
    title: "Prepare for funding",
    detail: "List the documents investors may expect",
    text: "What documents should I prepare before applying for startup funding?",
    domain: "funding_preparation",
  },
];

const conversation = computed(() => chatStore.activeMessages);
const isConsumerLoading = computed(() => loadingSessionId.value === chatStore.activeSessionId);
const isConsumerBusy = computed(() => isConsumerLoading.value);
const hasPendingDomainConfirmation = computed(
  () => pendingDomainConfirmation.value?.sessionId === chatStore.activeSessionId,
);
const hasPendingSensitiveReview = computed(() => pendingSensitiveReview.value !== null);
const sensitiveFindingLabels = computed(() =>
  pendingSensitiveReview.value?.findings.map((finding) => finding.label).join(", ") ?? "sensitive information",
);
const shouldShowTypingBubble = computed(() => isConsumerBusy.value && streamingMessageId.value === null);

const availableModels = computed(() => {
  if (status.value?.generation_models?.length) {
    return status.value.generation_models;
  }
  return status.value?.available_models ?? [];
});

const activeConsumerModel = computed(() => {
  if (consumerModel.value) return consumerModel.value;
  if (status.value?.configured_model) return status.value.configured_model;
  return availableModels.value[0] ?? "";
});

const activeConsumerModelDisplay = computed(() => {
  return activeConsumerModel.value || "Founder Assistant";
});

function syncModelSelections() {
  if (consumerModel.value && !availableModels.value.includes(consumerModel.value)) {
    consumerModel.value = "";
  }
}

async function refreshStatus() {
  try {
    status.value = await fetchLocalLLMStatus();
    syncModelSelections();
  } catch (err) {
    status.value = {
      reachable: false,
      error: err instanceof Error ? err.message : "Failed to check local LLM status.",
    };
  }
}

function scrollThreadToBottom() {
  nextTick(() => {
    if (!threadRef.value) return;
    threadRef.value.scrollTop = threadRef.value.scrollHeight;
  });
}

function resizeConsumerInput() {
  const target = consumerInputRef.value;
  if (!target) return;
  target.style.height = "0px";
  const next = Math.min(target.scrollHeight, 180);
  target.style.height = `${Math.max(next, 44)}px`;
}

function copyTextToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.select();

  try {
    const copied = document.execCommand("copy");
    if (!copied) {
      throw new Error("Copy command was rejected.");
    }
    return Promise.resolve();
  } finally {
    document.body.removeChild(textarea);
  }
}

async function copyAssistantMessage(messageId: number, content: string) {
  try {
    await copyTextToClipboard(content);
    copiedMessageId.value = messageId;
    if (copiedMessageTimer !== undefined) {
      window.clearTimeout(copiedMessageTimer);
    }
    copiedMessageTimer = window.setTimeout(() => {
      copiedMessageId.value = null;
      copiedMessageTimer = undefined;
    }, 1600);
  } catch (err) {
    consumerError.value = err instanceof Error ? err.message : "Failed to copy response.";
  }
}

function setAssistantFeedback(messageId: number, feedback: ConsumerChatFeedback) {
  chatStore.setMessageFeedback(chatStore.activeSessionId, messageId, feedback);
}

function formatMessageTime(value: string): string {
  const date = new Date(value);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatSourceDate(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const compactUtc = trimmed.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  const normalized = compactUtc
    ? `${compactUtc[1]}-${compactUtc[2]}-${compactUtc[3]}T${compactUtc[4]}:${compactUtc[5]}:${compactUtc[6]}Z`
    : trimmed;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return trimmed;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: /^\d{4}-?\d{2}-?\d{2}/.test(trimmed) ? "numeric" : undefined,
  }).format(date);
}

function formatCitationSourceName(value: string): string {
  return value
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/__\d{8}T\d{6}Z$/i, "")
    .replaceAll("__", " · ");
}

function isLegacyCollectionTimestamp(value: string | undefined): boolean {
  return /^\d{8}T\d{6}Z$/.test(value?.trim() ?? "");
}

function displayPublicationDate(item: RAGCitationItem): string {
  return isLegacyCollectionTimestamp(item.publication_date) ? "" : (item.publication_date ?? "");
}

function displayCollectionDate(item: RAGCitationItem): string {
  if (item.retrieved_at) return item.retrieved_at;
  return isLegacyCollectionTimestamp(item.publication_date) ? (item.publication_date ?? "") : "";
}

function formatLatencySeconds(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 s";
  const seconds = value / 1000;
  if (seconds < 10) return `${seconds.toFixed(1)} s`;
  return `${Math.round(seconds)} s`;
}

function normalizeModelResponse(text: string | undefined, context: "rag" | "llm"): string {
  const trimmed = (text ?? "").trim();
  if (trimmed) {
    return text ?? "";
  }
  if (context === "rag") {
    return "The selected model returned an empty RAG response. Please retry or switch to another model.";
  }
  return "The selected model returned an empty response. Please retry or switch to another model.";
}

function detectAnswerLanguage(text: string): AdvisorRequestAssessment["language"] {
  if (/\p{Script=Han}/u.test(text)) return "zh";
  const normalized = ` ${text.toLocaleLowerCase()} `;
  const germanMarkers = [" ich ", " der ", " die ", " das ", " welche ", " wie ", " für ", " gründ"];
  return germanMarkers.some((marker) => normalized.includes(marker)) ? "de" : "en";
}

function guardFallbackNotice(language: AdvisorRequestAssessment["language"]): string {
  if (language === "de") {
    return "> **Systemhinweis:** Die erweiterte Anfrageprüfung ist vorübergehend nicht verfügbar. Grundlegende Sicherheitsregeln wurden angewendet.";
  }
  if (language === "zh") {
    return "> **系统提示：** 高级请求检查暂时不可用，本次请求已使用基础安全规则进行检查。";
  }
  return "> **System notice:** Advanced request assessment is temporarily unavailable. Basic safety rules were applied.";
}

function modelServiceUnavailableMessage(language: AdvisorRequestAssessment["language"]): string {
  if (language === "de") {
    return "Der Modelldienst ist derzeit nicht verfügbar oder ausgelastet. Bitte versuchen Sie es später erneut oder wählen Sie ein anderes verfügbares Modell.";
  }
  if (language === "zh") {
    return "模型服务当前不可用或正处于繁忙状态。请稍后重试，或选择其他可用模型。";
  }
  return "The model service is currently unavailable or busy. Please try again later or select another available model.";
}

function interruptedResponseMessage(language: AdvisorRequestAssessment["language"]): string {
  if (language === "de") {
    return "_Die Modellantwort wurde vor dem regulären Abschluss unterbrochen. Bitte versuchen Sie es erneut oder wählen Sie ein anderes Modell._";
  }
  if (language === "zh") {
    return "_模型回答在正常完成前中断。请重试或选择其他模型。_";
  }
  return "_The model response ended before normal completion. Please retry or select another model._";
}

function tokenLimitResponseMessage(language: AdvisorRequestAssessment["language"]): string {
  if (language === "de") {
    return "_Die Antwort wurde am Ausgabelimit abgeschnitten. Bitte versuchen Sie es erneut oder bitten Sie um eine kürzere Antwort._";
  }
  if (language === "zh") {
    return "_回答因达到输出长度上限而被截断。请重试，或要求模型给出更简短的回答。_";
  }
  return "_The response was cut off at the output limit. Please retry or ask for a shorter answer._";
}

function onConsumerInputKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (
      !isConsumerBusy.value &&
      !hasPendingDomainConfirmation.value &&
      !hasPendingSensitiveReview.value &&
      consumerInput.value.trim()
    ) {
      void sendConsumerPrompt();
    }
  }
}

function readStoredConsumerModel(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(CONSUMER_MODEL_STORAGE_KEY) ?? "";
}

function writeStoredConsumerModel(value: string) {
  if (typeof window === "undefined") return;
  if (!value) {
    window.localStorage.removeItem(CONSUMER_MODEL_STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(CONSUMER_MODEL_STORAGE_KEY, value);
}

function startGenerationTimer() {
  generationStartedAt.value = Date.now();
  generationElapsedMs.value = 0;
  if (generationTimer !== undefined) {
    window.clearInterval(generationTimer);
  }
  generationTimer = window.setInterval(() => {
    if (generationStartedAt.value === null) return;
    generationElapsedMs.value = Date.now() - generationStartedAt.value;
  }, 200);
}

function stopGenerationTimer() {
  if (generationTimer !== undefined) {
    window.clearInterval(generationTimer);
    generationTimer = undefined;
  }
  generationStartedAt.value = null;
}

function estimateHistoryTokens(text: string): number {
  const normalized = text.trim().replace(/\s+/g, " ");
  if (!normalized) return 0;
  let weightedUnits = 0;
  for (const character of normalized) {
    weightedUnits += (character.codePointAt(0) ?? 0) < 128 ? 1 : 4;
  }
  return Math.max(1, Math.ceil(weightedUnits / 4));
}

function truncateHistoryText(text: string, maxTokens: number): string {
  const normalized = text.trim().replace(/\s+/g, " ");
  if (!normalized || maxTokens < 1) return "";
  if (estimateHistoryTokens(normalized) <= maxTokens) return normalized;

  const unitBudget = maxTokens * 4 - 4;
  let usedUnits = 0;
  const characters: string[] = [];
  for (const character of normalized) {
    const characterUnits = (character.codePointAt(0) ?? 0) < 128 ? 1 : 4;
    if (usedUnits + characterUnits > unitBudget) break;
    characters.push(character);
    usedUnits += characterUnits;
  }
  return characters.length ? `${characters.join("").trimEnd()}...` : "";
}

function buildSessionConversationHistory(sessionId: string): RAGConversationMessage[] {
  const candidates = (chatStore.messagesBySession[sessionId] ?? [])
    .filter((message) => !message.error && !message.streaming && message.content.trim())
    .slice(-(MAX_HISTORY_TURNS * 2));
  const selected: RAGConversationMessage[] = [];
  let usedTokens = 0;

  for (const message of [...candidates].reverse()) {
    const remainingTokens = MAX_HISTORY_TOKENS - usedTokens;
    const messageBudget = Math.min(MAX_HISTORY_MESSAGE_TOKENS, remainingTokens - 4);
    if (messageBudget < 1) break;
    const content = truncateHistoryText(message.content, messageBudget);
    if (!content) continue;
    selected.unshift({ role: message.role, content });
    usedTokens += estimateHistoryTokens(content) + 4;
  }

  return selected;
}

async function sendConsumerPrompt(seedPrompt?: string, sensitiveReviewAccepted = false) {
  if (isConsumerBusy.value) return;

  const text = (seedPrompt ?? consumerInput.value).trim();
  if (!text) return;
  if (!sensitiveReviewAccepted) {
    const findings = detectStructuredSensitiveInput(text);
    if (findings.length) {
      pendingSensitiveReview.value = { text, findings };
      return;
    }
  }
  const targetSessionId = chatStore.activeSessionId;
  const history = buildSessionConversationHistory(targetSessionId);

  // A preselected topic is an explicit user choice, so it can immediately use
  // the evaluated domain -> Dense@5 retrieval path.
  if (consumerDomain.value !== "ask") {
    await startConsumerRAGAnswer(
      text,
      consumerDomain.value === "auto" ? undefined : consumerDomain.value,
      true,
      history,
    );
    return;
  }

  consumerError.value = "";
  chatStore.addMessage(
    {
      role: "user",
      content: text,
      model: "",
      latencyMs: 0,
      retrievalLatencyMs: 0,
      citations: [],
      error: false,
    },
    targetSessionId,
  );
  consumerInput.value = "";
  resizeConsumerInput();
  pendingDomainConfirmation.value = {
    sessionId: targetSessionId,
    prompt: text,
    history,
  };
  scrollThreadToBottom();
}

function editSensitiveInput() {
  pendingSensitiveReview.value = null;
  nextTick(() => consumerInputRef.value?.focus());
}

function continueWithSensitiveInput() {
  const pending = pendingSensitiveReview.value;
  if (!pending) return;
  pendingSensitiveReview.value = null;
  void sendConsumerPrompt(pending.text, true);
}

async function confirmPendingDomain(domain?: RAGDecisionDomain) {
  const pending = pendingDomainConfirmation.value;
  if (!pending || pending.sessionId !== chatStore.activeSessionId || isConsumerBusy.value) return;

  pendingDomainConfirmation.value = null;
  await startConsumerRAGAnswer(pending.prompt, domain, false, pending.history);
}

async function startConsumerRAGAnswer(
  text: string,
  domain: RAGDecisionDomain | undefined,
  addUserMessage = true,
  history?: RAGConversationMessage[],
) {
  if (loadingSessionId.value) return;

  const targetSessionId = chatStore.activeSessionId;
  const conversationHistory = history ?? buildSessionConversationHistory(targetSessionId);
  let assistantMessageId: number | null = null;
  let streamedText = "";
  let streamRetrievalLatencyMs = 0;
  let streamCitations: RAGCitationItem[] = [];
  let streamRuntimeProfile: RAGRuntimeProfile | undefined;
  let requestAssessment: AdvisorRequestAssessment | null = null;
  let failureStage = "request_assessment";
  consumerError.value = "";
  chatStore.setSessionDomain(targetSessionId, domain ?? "auto");
  if (addUserMessage) {
    chatStore.addMessage(
      {
        role: "user",
        content: text,
        model: "",
        latencyMs: 0,
        retrievalLatencyMs: 0,
        citations: [],
        error: false,
      },
      targetSessionId,
    );
  }

  consumerInput.value = "";
  resizeConsumerInput();
  loadingSessionId.value = targetSessionId;
  startGenerationTimer();
  scrollThreadToBottom();

  try {
    requestAssessment = await assessAdvisorRequest({
      prompt: text,
      domain,
      history: conversationHistory,
      model: activeConsumerModel.value || undefined,
    });

    if (requestAssessment.action === "clarify" || requestAssessment.action === "refuse_and_redirect") {
      assistantMessageId = chatStore.addMessage(
        {
          role: "assistant",
          content: requestAssessment.message,
          model: activeConsumerModel.value,
          latencyMs: generationElapsedMs.value,
          retrievalLatencyMs: 0,
          citations: [],
          error: false,
          requestAssessment,
          streaming: false,
        },
        targetSessionId,
      );
      scrollThreadToBottom();
      return;
    }

    const answerLanguage = requestAssessment.language;
    const assessmentNotices = [
      requestAssessment.source === "fallback_rule"
        ? guardFallbackNotice(requestAssessment.language)
        : "",
      requestAssessment.notice,
    ].filter(Boolean);
    const assessmentNotice = assessmentNotices.length
      ? `${assessmentNotices.join("\n\n")}\n\n`
      : "";
    const guardedSystemPrompt = requestAssessment.generation_instruction
      ? `${systemPrompt.value}\n\n${requestAssessment.generation_instruction}`
      : systemPrompt.value;
    assistantMessageId = chatStore.addMessage(
      {
        role: "assistant",
        content: assessmentNotice,
        model: activeConsumerModel.value,
        latencyMs: 0,
        retrievalLatencyMs: 0,
        citations: [],
        error: false,
        requestAssessment,
        streaming: true,
      },
      targetSessionId,
    );
    streamingMessageId.value = assistantMessageId;
    failureStage = "answer_generation";
    scrollThreadToBottom();

    try {
      await streamChatWithRAG(
        {
          prompt: text,
          system_prompt: guardedSystemPrompt,
          temperature: 0.2,
          max_tokens: TEMPORARY_CONSUMER_MAX_TOKENS,
          domain,
          history: conversationHistory,
          model: activeConsumerModel.value || undefined,
        },
        {
          onProfile: (profile) => {
            streamRuntimeProfile = profile;
            if (assistantMessageId !== null) {
              chatStore.updateMessage(targetSessionId, assistantMessageId, { runtimeProfile: profile }, false);
            }
          },
          onMetadata: (metadata) => {
            streamRetrievalLatencyMs = metadata.retrieval_latency_ms;
            streamCitations = metadata.citations;
            if (assistantMessageId !== null) {
              chatStore.updateMessage(
                targetSessionId,
                assistantMessageId,
                {
                  retrievalLatencyMs: metadata.retrieval_latency_ms,
                },
                false,
              );
            }
          },
          onToken: (token) => {
            streamedText += token;
            if (assistantMessageId !== null) {
              chatStore.appendMessageContent(targetSessionId, assistantMessageId, token);
            }
            scrollThreadToBottom();
          },
          onDone: (done) => {
            if (assistantMessageId !== null) {
              const hitTokenLimit = done.finish_reason === "length";
              const generationIncomplete = !done.done || hitTokenLimit;
              if (generationIncomplete) {
                chatStore.appendMessageContent(
                  targetSessionId,
                  assistantMessageId,
                  `\n\n${
                    hitTokenLimit
                      ? tokenLimitResponseMessage(answerLanguage)
                      : interruptedResponseMessage(answerLanguage)
                  }`,
                );
              }
              chatStore.updateMessage(targetSessionId, assistantMessageId, {
                model: done.model,
                latencyMs: done.latency_ms,
                retrievalLatencyMs: streamRetrievalLatencyMs,
                citations: streamCitations,
                runtimeProfile: streamRuntimeProfile,
                streaming: false,
                error: generationIncomplete,
                generationCompleted: done.done && !hitTokenLimit,
                finishReason: done.finish_reason ?? "",
              });
            }
          },
        },
      );

      if (!streamedText.trim()) {
        const baseline = await chatWithLocalLLM({
          prompt: text,
          system_prompt: guardedSystemPrompt,
          temperature: 0.2,
          max_tokens: TEMPORARY_CONSUMER_MAX_TOKENS,
          model: activeConsumerModel.value || undefined,
        });

        if (assistantMessageId !== null) {
          chatStore.updateMessage(targetSessionId, assistantMessageId, {
            content:
              assessmentNotice +
              "The selected model returned an empty RAG response. Showing fallback answer from base LLM:\n\n" +
              normalizeModelResponse(baseline.response, "llm"),
            model: baseline.model || activeConsumerModel.value,
            latencyMs: baseline.latency_ms,
            retrievalLatencyMs: streamRetrievalLatencyMs,
            citations: streamCitations,
            error: false,
            runtimeProfile: streamRuntimeProfile,
            streaming: false,
          });
        }
        return;
      }
    } catch (ragErr) {
      if (streamedText.trim()) {
        const technicalError = ragErr instanceof Error ? ragErr.message : "request failed";
        console.error("[advisor-request-failed]", {
          stage: "streaming_answer",
          model: activeConsumerModel.value,
          error: technicalError,
        });
        if (assistantMessageId !== null) {
          chatStore.updateMessage(targetSessionId, assistantMessageId, {
            content:
              assessmentNotice +
              streamedText +
              "\n\n" +
              interruptedResponseMessage(requestAssessment.language),
            retrievalLatencyMs: streamRetrievalLatencyMs,
            citations: streamCitations,
            error: true,
            streaming: false,
          });
        }
        return;
      }

      const baseline = await chatWithLocalLLM({
        prompt: text,
        system_prompt: guardedSystemPrompt,
        temperature: 0.2,
        max_tokens: TEMPORARY_CONSUMER_MAX_TOKENS,
        model: activeConsumerModel.value || undefined,
      });

      if (assistantMessageId !== null) {
        console.warn("[advisor-rag-fallback]", {
          stage: "rag_answer",
          model: activeConsumerModel.value,
          error: ragErr instanceof Error ? ragErr.message : "request failed",
        });
        chatStore.updateMessage(targetSessionId, assistantMessageId, {
          content:
            assessmentNotice +
            "Knowledge retrieval is temporarily unavailable. Falling back to base LLM response.\n\n" +
            normalizeModelResponse(baseline.response, "llm"),
          model: baseline.model,
          latencyMs: baseline.latency_ms,
          retrievalLatencyMs: 0,
          citations: [],
          error: false,
          streaming: false,
        });
      }
    }
  } catch (err) {
    const technicalError = err instanceof Error ? err.message : "Request failed.";
    const answerLanguage = requestAssessment?.language ?? detectAnswerLanguage(text);
    const userMessage = modelServiceUnavailableMessage(answerLanguage);
    console.error("[advisor-request-failed]", {
      stage: failureStage,
      model: activeConsumerModel.value,
      guardSource: requestAssessment?.source ?? "unavailable",
      error: technicalError,
    });
    consumerError.value = userMessage;
    if (assistantMessageId !== null) {
      chatStore.updateMessage(targetSessionId, assistantMessageId, {
        content: userMessage,
        model: "",
        latencyMs: 0,
        retrievalLatencyMs: 0,
        citations: [],
        error: true,
        streaming: false,
      });
    } else {
      chatStore.addMessage(
        {
          role: "assistant",
          content: userMessage,
          model: "",
          latencyMs: 0,
          retrievalLatencyMs: 0,
          citations: [],
          error: true,
        },
        targetSessionId,
      );
    }
  } finally {
    if (loadingSessionId.value === targetSessionId) {
      loadingSessionId.value = null;
      streamingMessageId.value = null;
      stopGenerationTimer();
    }
    scrollThreadToBottom();
  }
}

function submitConsumerForm() {
  void sendConsumerPrompt();
}

function sendStarterPrompt(item: StarterPrompt) {
  void startConsumerRAGAnswer(item.text, item.domain);
}

onMounted(() => {
  void refreshStatus();
  resizeConsumerInput();
});

onBeforeUnmount(() => {
  stopGenerationTimer();
  if (copiedMessageTimer !== undefined) {
    window.clearTimeout(copiedMessageTimer);
  }
});

watch(
  () => chatStore.activeSessionId,
  () => {
    consumerError.value = "";
    consumerInput.value = "";
    pendingSensitiveReview.value = null;
    resizeConsumerInput();
    if (loadingSessionId.value && loadingSessionId.value !== chatStore.activeSessionId) {
      consumerError.value = "Another chat is still generating. Please wait a moment.";
    }
    scrollThreadToBottom();
  },
);

watch(
  () => conversation.value.length,
  () => {
    scrollThreadToBottom();
  },
);

watch(consumerInput, () => {
  resizeConsumerInput();
}, { flush: "post" });

watch(consumerModel, (value) => {
  writeStoredConsumerModel(value);
});
</script>
