import { defineStore } from "pinia";

import type {
  AdvisorRequestAssessment,
  RAGCitationItem,
  RAGDecisionDomain,
  RAGRuntimeProfile,
} from "@/services/localLlm";

export type ConsumerChatFeedback = "up" | "down";

export interface ConsumerChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  model: string;
  latencyMs: number;
  retrievalLatencyMs: number;
  citations: RAGCitationItem[];
  error: boolean;
  createdAt: string;
  feedback?: ConsumerChatFeedback;
  runtimeProfile?: RAGRuntimeProfile;
  requestAssessment?: AdvisorRequestAssessment;
  streaming?: boolean;
  generationCompleted?: boolean;
  finishReason?: string;
}

export interface ConsumerChatSession {
  id: string;
  title: string;
  lastMessagePreview: string;
  updatedAt: string;
  domain?: RAGDecisionDomain | "auto";
}

interface ConsumerChatState {
  sessions: ConsumerChatSession[];
  activeSessionId: string;
  messagesBySession: Record<string, ConsumerChatMessage[]>;
}

const CHAT_STORAGE_KEY = "founder-assistant-consumer-chat-v1";
const MAX_SESSIONS = 40;
const VALID_RAG_DOMAINS = new Set<RAGDecisionDomain | "auto">([
  "auto",
  "legal_form_selection",
  "funding_preparation",
  "market_validation",
]);

function createSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

function sanitizeSessionTitle(raw: string): string {
  const trimmed = raw.trim().replace(/\s+/g, " ");
  if (!trimmed) return "New chat";
  return trimmed.length <= 40 ? trimmed : `${trimmed.slice(0, 40)}...`;
}

function safeParseState(raw: string | null): ConsumerChatState | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as ConsumerChatState;
    if (!parsed || typeof parsed !== "object") return null;
    if (!Array.isArray(parsed.sessions) || !parsed.activeSessionId || !parsed.messagesBySession) return null;
    for (const session of parsed.sessions) {
      if (session.domain && !VALID_RAG_DOMAINS.has(session.domain)) {
        delete session.domain;
      }
    }
    for (const list of Object.values(parsed.messagesBySession)) {
      for (const message of list) {
        if (message.streaming) {
          message.streaming = false;
          message.error = true;
          if (!message.content.trim()) {
            message.content = "Generation was interrupted before completion.";
          }
        }
      }
    }
    return parsed;
  } catch {
    return null;
  }
}

function writeStateToStorage(state: ConsumerChatState): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(state));
}

function buildInitialState(): ConsumerChatState {
  if (typeof window !== "undefined") {
    const parsed = safeParseState(window.localStorage.getItem(CHAT_STORAGE_KEY));
    if (parsed) {
      return parsed;
    }
  }

  const sessionId = createSessionId();
  const createdAt = nowIso();
  return {
    sessions: [
      {
        id: sessionId,
        title: "New chat",
        lastMessagePreview: "",
        updatedAt: createdAt,
      },
    ],
    activeSessionId: sessionId,
    messagesBySession: {
      [sessionId]: [],
    },
  };
}

export const useConsumerChatStore = defineStore("consumer-chat", {
  state: () => buildInitialState(),
  getters: {
    activeSession(state): ConsumerChatSession | null {
      return state.sessions.find((item) => item.id === state.activeSessionId) ?? null;
    },
    activeMessages(state): ConsumerChatMessage[] {
      return state.messagesBySession[state.activeSessionId] ?? [];
    },
  },
  actions: {
    persist() {
      writeStateToStorage({
        sessions: this.sessions,
        activeSessionId: this.activeSessionId,
        messagesBySession: this.messagesBySession,
      });
    },

    ensureSession() {
      if (this.sessions.length > 0 && this.activeSessionId) return;
      const sessionId = createSessionId();
      const createdAt = nowIso();
      this.sessions = [
        {
          id: sessionId,
          title: "New chat",
          lastMessagePreview: "",
          updatedAt: createdAt,
        },
      ];
      this.activeSessionId = sessionId;
      this.messagesBySession = {
        [sessionId]: [],
      };
      this.persist();
    },

    startNewSession() {
      const sessionId = createSessionId();
      const createdAt = nowIso();
      this.sessions.unshift({
        id: sessionId,
        title: "New chat",
        lastMessagePreview: "",
        updatedAt: createdAt,
      });

      if (this.sessions.length > MAX_SESSIONS) {
        const removed = this.sessions.splice(MAX_SESSIONS);
        for (const item of removed) {
          delete this.messagesBySession[item.id];
        }
      }

      this.messagesBySession[sessionId] = [];
      this.activeSessionId = sessionId;
      this.persist();
    },

    selectSession(sessionId: string) {
      if (!this.messagesBySession[sessionId]) return;
      this.activeSessionId = sessionId;
      this.persist();
    },

    renameSession(sessionId: string, title: string) {
      const target = this.sessions.find((item) => item.id === sessionId);
      if (!target) return;
      target.title = sanitizeSessionTitle(title);
      this.persist();
    },

    setSessionDomain(sessionId: string, domain?: RAGDecisionDomain | "auto") {
      const target = this.sessions.find((item) => item.id === sessionId);
      if (!target) return;
      if (domain) {
        target.domain = domain;
      } else {
        delete target.domain;
      }
      this.persist();
    },

    deleteSession(sessionId: string) {
      const targetIndex = this.sessions.findIndex((item) => item.id === sessionId);
      if (targetIndex === -1) return;

      this.sessions.splice(targetIndex, 1);
      delete this.messagesBySession[sessionId];

      if (this.sessions.length === 0) {
        const nextSessionId = createSessionId();
        const createdAt = nowIso();
        this.sessions = [
          {
            id: nextSessionId,
            title: "New chat",
            lastMessagePreview: "",
            updatedAt: createdAt,
          },
        ];
        this.messagesBySession = {
          [nextSessionId]: [],
        };
        this.activeSessionId = nextSessionId;
        this.persist();
        return;
      }

      if (this.activeSessionId === sessionId) {
        const nextSession = this.sessions[Math.min(targetIndex, this.sessions.length - 1)];
        this.activeSessionId = nextSession.id;
      }

      this.persist();
    },

    setMessageFeedback(sessionId: string, messageId: number, feedback: ConsumerChatFeedback) {
      const list = this.messagesBySession[sessionId];
      if (!list) return;

      const target = list.find((item) => item.id === messageId && item.role === "assistant");
      if (!target) return;

      target.feedback = target.feedback === feedback ? undefined : feedback;
      this.persist();
    },

    addMessage(message: Omit<ConsumerChatMessage, "id" | "createdAt">, sessionId?: string): number | null {
      this.ensureSession();
      const targetSessionId = sessionId ?? this.activeSessionId;
      if (!this.sessions.some((item) => item.id === targetSessionId)) return null;
      if (!this.messagesBySession[targetSessionId]) {
        this.messagesBySession[targetSessionId] = [];
      }

      const list = this.messagesBySession[targetSessionId] ?? [];
      const nextMessage: ConsumerChatMessage = {
        id: Date.now() + Math.floor(Math.random() * 1000),
        createdAt: nowIso(),
        ...message,
      };
      list.push(nextMessage);
      this.messagesBySession[targetSessionId] = list;

      const target = this.sessions.find((item) => item.id === targetSessionId);
      if (target) {
        if (target.title === "New chat" && message.role === "user" && list.length === 1) {
          target.title = sanitizeSessionTitle(message.content);
        }
        target.lastMessagePreview = message.content.slice(0, 70);
        target.updatedAt = nowIso();
      }

      this.sessions.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1));
      this.persist();
      return nextMessage.id;
    },

    appendMessageContent(sessionId: string, messageId: number, delta: string) {
      const list = this.messagesBySession[sessionId];
      if (!list || !delta) return;

      const target = list.find((item) => item.id === messageId);
      if (!target) return;

      target.content += delta;
    },

    updateMessage(
      sessionId: string,
      messageId: number,
      patch: Partial<Omit<ConsumerChatMessage, "id" | "createdAt" | "role">>,
      persist = true,
    ) {
      const list = this.messagesBySession[sessionId];
      if (!list) return;

      const target = list.find((item) => item.id === messageId);
      if (!target) return;

      Object.assign(target, patch);
      const session = this.sessions.find((item) => item.id === sessionId);
      if (session) {
        session.lastMessagePreview = target.content.slice(0, 70);
        session.updatedAt = nowIso();
      }

      if (persist) {
        this.persist();
      }
    },

    clearActiveSessionMessages() {
      this.ensureSession();
      this.messagesBySession[this.activeSessionId] = [];
      const session = this.sessions.find((item) => item.id === this.activeSessionId);
      if (session) {
        session.title = "New chat";
        session.lastMessagePreview = "";
        session.updatedAt = nowIso();
        delete session.domain;
      }
      this.persist();
    },
  },
});
