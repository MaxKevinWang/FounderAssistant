<template>
  <div class="app-shell mode-consumer" :class="{ 'sidebar-open': sidebarOpen }">
    <aside class="app-sidebar">
      <div class="sidebar-main">
        <div class="brand-block">
          <p class="brand-kicker">Founder Copilot</p>
          <h1>Founder Assistant</h1>
        </div>

        <button class="new-chat-btn" type="button" @click="startNewChat">+ New chat</button>

        <div class="session-list">
          <p class="session-list-label">Recent chats</p>
          <div
            v-for="item in chatStore.sessions"
            :key="item.id"
            class="session-item"
            :class="{ active: item.id === chatStore.activeSessionId }"
          >
            <template v-if="editingSessionId === item.id">
              <input
                v-model="editingSessionTitle"
                class="session-title-input"
                :data-session-title-input="item.id"
                type="text"
                aria-label="Rename chat"
                @keydown.enter.prevent="saveRenamedSession(item.id)"
                @keydown.esc.prevent="cancelRenameSession"
              />
              <div class="session-edit-actions">
                <button class="session-action-btn" type="button" @click="saveRenamedSession(item.id)">Save</button>
                <button class="session-action-btn" type="button" @click="cancelRenameSession">Cancel</button>
              </div>
            </template>

            <template v-else>
              <button class="session-open-btn" type="button" @click="openSession(item.id)">
                <span class="session-item-title">{{ item.title }}</span>
                <span class="session-item-meta">
                  <span class="session-item-time">{{ formatSessionTime(item.updatedAt) }}</span>
                  <span v-if="item.lastMessagePreview">{{ item.lastMessagePreview }}</span>
                </span>
              </button>
              <div class="session-menu-wrap">
                <button
                  class="session-menu-trigger"
                  type="button"
                  aria-label="Chat actions"
                  :aria-expanded="openSessionMenuId === item.id"
                  @click.stop="toggleSessionMenu(item.id, $event)"
                >
                  <span class="session-menu-dots" aria-hidden="true">...</span>
                </button>
                <div
                  v-if="openSessionMenuId === item.id"
                  class="session-context-menu"
                  role="menu"
                  :style="{ top: `${sessionMenuPosition.top}px`, left: `${sessionMenuPosition.left}px` }"
                >
                  <button class="session-menu-option" type="button" role="menuitem" @click="startRenameSession(item.id, item.title)">
                    <svg class="session-menu-icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M4 20h4.2L18.7 9.5l-4.2-4.2L4 15.8V20Z" />
                      <path d="m16 3.8 4.2 4.2" />
                    </svg>
                    <span>Rename</span>
                  </button>
                  <template v-if="pendingDeleteSessionId === item.id">
                    <div class="session-delete-confirm" role="group" aria-label="Confirm delete chat">
                      <p class="session-delete-title">Delete this chat?</p>
                      <p class="session-delete-copy">This cannot be undone.</p>
                      <div class="session-delete-actions">
                        <button class="session-confirm-btn" type="button" @click="cancelDeleteSession">Cancel</button>
                        <button class="session-confirm-btn danger" type="button" @click="confirmDeleteSession(item.id)">Delete</button>
                      </div>
                    </div>
                  </template>
                  <button v-else class="session-menu-option danger" type="button" role="menuitem" @click="startDeleteSession(item.id)">
                    <svg class="session-menu-icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M5 7h14" />
                      <path d="M10 11v6" />
                      <path d="M14 11v6" />
                      <path d="M7 7l1 13h8l1-13" />
                      <path d="M9 7V4h6v3" />
                    </svg>
                    <span>Delete</span>
                  </button>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </aside>

    <button class="mobile-sidebar-mask" type="button" aria-label="Close sidebar" @click="sidebarOpen = false"></button>

    <div class="main-area">
      <button class="consumer-sidebar-toggle" type="button" aria-label="Open chats" @click="sidebarOpen = true">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <main class="content-panel">
        <RouterView />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";

import { useConsumerChatStore } from "@/stores/consumerChat";

const chatStore = useConsumerChatStore();
const route = useRoute();
const router = useRouter();

const sidebarOpen = ref(false);
const editingSessionId = ref("");
const editingSessionTitle = ref("");
const openSessionMenuId = ref("");
const pendingDeleteSessionId = ref("");
const sessionMenuPosition = ref({ top: 0, left: 0 });

function formatSessionTime(value: string): string {
  const date = new Date(value);
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  return sameDay
    ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function startNewChat() {
  cancelRenameSession();
  closeSessionMenu();
  chatStore.startNewSession();
  if (route.name !== "advisor") void router.push({ name: "advisor" });
  sidebarOpen.value = false;
}

function openSession(sessionId: string) {
  cancelRenameSession();
  closeSessionMenu();
  chatStore.selectSession(sessionId);
  if (route.name !== "advisor") void router.push({ name: "advisor" });
  sidebarOpen.value = false;
}

function toggleSessionMenu(sessionId: string, event: MouseEvent) {
  cancelRenameSession();
  if (openSessionMenuId.value === sessionId) {
    closeSessionMenu();
    return;
  }
  const target = event.currentTarget;
  if (target instanceof HTMLElement) {
    const rect = target.getBoundingClientRect();
    const menuWidth = 168;
    const viewportPadding = 8;
    sessionMenuPosition.value = {
      top: rect.bottom + 6,
      left: Math.min(window.innerWidth - menuWidth - viewportPadding, Math.max(viewportPadding, rect.right - menuWidth)),
    };
  }
  openSessionMenuId.value = sessionId;
  pendingDeleteSessionId.value = "";
}

function closeSessionMenu() {
  openSessionMenuId.value = "";
  pendingDeleteSessionId.value = "";
}

function startRenameSession(sessionId: string, currentTitle: string) {
  closeSessionMenu();
  editingSessionId.value = sessionId;
  editingSessionTitle.value = currentTitle;
  void nextTick(() => {
    const input = document.querySelector<HTMLInputElement>(`[data-session-title-input="${sessionId}"]`);
    input?.focus();
    input?.select();
  });
}

function saveRenamedSession(sessionId: string) {
  if (editingSessionId.value !== sessionId) return;
  chatStore.renameSession(sessionId, editingSessionTitle.value);
  cancelRenameSession();
}

function cancelRenameSession() {
  editingSessionId.value = "";
  editingSessionTitle.value = "";
}

function startDeleteSession(sessionId: string) {
  pendingDeleteSessionId.value = sessionId;
}

function cancelDeleteSession() {
  pendingDeleteSessionId.value = "";
}

function confirmDeleteSession(sessionId: string) {
  closeSessionMenu();
  if (editingSessionId.value === sessionId) cancelRenameSession();
  chatStore.deleteSession(sessionId);
}

function handleGlobalPointerDown(event: MouseEvent) {
  const target = event.target;
  if (!(target instanceof HTMLElement) || target.closest(".session-menu-wrap")) return;
  closeSessionMenu();
}

onMounted(() => {
  chatStore.ensureSession();
  window.addEventListener("pointerdown", handleGlobalPointerDown);
});

onBeforeUnmount(() => {
  window.removeEventListener("pointerdown", handleGlobalPointerDown);
});
</script>
