import { createRouter, createWebHistory } from "vue-router";

import AdvisorView from "@/views/AdvisorView.vue";

// The production application exposes only the consumer advisor.
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/advisor" },
    { path: "/advisor", name: "advisor", component: AdvisorView },
    { path: "/:pathMatch(.*)*", redirect: "/advisor" },
  ],
});

export default router;
