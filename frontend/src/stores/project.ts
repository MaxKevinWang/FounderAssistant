import { defineStore } from "pinia";

interface DecisionDomain {
  id: string;
  title: string;
  focus: string;
  keyInputs: string[];
  outputTemplate: string[];
}

interface ComparisonMetric {
  metric: string;
  baseline: string;
  rag: string;
  interpretation: string;
}

interface ScenarioItem {
  code: string;
  domain: string;
  title: string;
  status: "planned" | "running" | "done";
  notes: string;
}

interface TrueFalseQuestion {
  id: string;
  statement: string;
  label: "TRUE" | "FALSE";
  source: string;
}

export const useProjectStore = defineStore("project", {
  state: () => ({
    thesisTitle: "Design and Evaluation of an Open-Source LLM- and RAG-based Decision Support System",
    method: "Design Science Research (DSR)",
    mainQuestion:
      "How can an open-source LLM- and RAG-based decision support system be designed for early-stage entrepreneurs?",
    researchQuestions: [
      "RQ1: Information needs and decision support requirements in selected domains",
      "RQ2: Domain knowledge structuring and integration into RAG",
      "RQ3: Improvement in relevance, traceability, and perceived usefulness vs baseline",
    ],
    decisionDomains: [
      {
        id: "legal_form",
        title: "Legal Form Selection",
        focus: "Choose suitable legal entity under liability, capital, and growth constraints",
        keyInputs: ["Founder count", "Initial capital", "Liability tolerance", "Investment plans"],
        outputTemplate: ["Recommended option", "Reasoning", "Risks", "Next legal steps"],
      },
      {
        id: "funding",
        title: "Funding Preparation",
        focus: "Prepare for grants, angels, or accelerator funding pathways",
        keyInputs: ["Stage", "Runway", "Target amount", "Current traction"],
        outputTemplate: ["Funding roadmap", "Required documents", "Milestones", "Risk checklist"],
      },
      {
        id: "market_validation",
        title: "Market Validation",
        focus: "Test assumptions before full product commitment",
        keyInputs: ["Target segment", "Hypothesis", "Time/budget", "Validation channel"],
        outputTemplate: ["Experiment design", "Metrics", "Thresholds", "Pivot/persevere rule"],
      },
    ] as DecisionDomain[],
    comparisonMetrics: [
      {
        metric: "Relevance",
        baseline: "3.0 / 5",
        rag: "4.2 / 5",
        interpretation: "RAG answers are more aligned with domain task context",
      },
      {
        metric: "Traceability",
        baseline: "1.8 / 5",
        rag: "4.6 / 5",
        interpretation: "RAG provides explicit evidence and source references",
      },
      {
        metric: "Perceived Usefulness",
        baseline: "3.1 / 5",
        rag: "4.3 / 5",
        interpretation: "Users rate RAG advice as more actionable",
      },
    ] as ComparisonMetric[],
    scenarios: [
      { code: "LF-001", domain: "Legal Form", title: "Solo founder with low capital", status: "done", notes: "UG vs GmbH decision path generated" },
      { code: "LF-002", domain: "Legal Form", title: "Two co-founders, investor-ready setup", status: "running", notes: "Governance and signaling criteria under review" },
      { code: "FP-001", domain: "Funding", title: "Angel pitch preparation", status: "done", notes: "Checklist and milestones validated" },
      { code: "FP-002", domain: "Funding", title: "Grant + accelerator strategy", status: "planned", notes: "Waiting for source updates" },
      { code: "MV-001", domain: "Market Validation", title: "Interview-based fit testing", status: "done", notes: "Experiment protocol completed" },
      { code: "MV-002", domain: "Market Validation", title: "Landing-page MVP test", status: "running", notes: "Metrics threshold tuning" },
    ] as ScenarioItem[],
    trueFalseQuestions: [
      {
        id: "TF-01",
        statement:
          "In GruenderZeiten 11, legal minimum capital is explicitly required only for GmbH, UG (haftungsbeschraenkt), and AG.",
        label: "TRUE",
        source: "GruenderZeiten 11",
      },
      {
        id: "TF-02",
        statement: "GruenderZeiten 11 states that a UG (haftungsbeschraenkt) requires at least EUR 25,000 minimum capital.",
        label: "FALSE",
        source: "GruenderZeiten 11",
      },
      {
        id: "TF-03",
        statement: "GruenderZeiten 24 says there are no permit-based trades at all in Germany.",
        label: "FALSE",
        source: "GruenderZeiten 24",
      },
      {
        id: "TF-04",
        statement:
          "GruenderZeiten 24 states that founders must generally announce the start of self-employment, whether full-time or part-time.",
        label: "TRUE",
        source: "GruenderZeiten 24",
      },
      {
        id: "TF-05",
        statement:
          "GruenderZeiten 24 says freelancers should first go to the Gewerbeamt, while trade businesses should first go to the Finanzamt.",
        label: "FALSE",
        source: "GruenderZeiten 24",
      },
      {
        id: "TF-06",
        statement:
          "GruenderZeiten 09 says founders must register with the Finanzamt within one month (rule introduced from 1 Jan 2021).",
        label: "TRUE",
        source: "GruenderZeiten 09",
      },
      {
        id: "TF-07",
        statement:
          "GruenderZeiten 09 says the Fragebogen zur steuerlichen Erfassung should be submitted electronically via ELSTER.",
        label: "TRUE",
        source: "GruenderZeiten 09",
      },
      {
        id: "TF-08",
        statement: "GruenderZeiten 09 says there are special founder-only tax privileges in Germany.",
        label: "FALSE",
        source: "GruenderZeiten 09",
      },
      {
        id: "TF-09",
        statement:
          "GruenderZeiten 24 states that for internet distance sales, consumers generally have a 14-day withdrawal period.",
        label: "TRUE",
        source: "GruenderZeiten 24",
      },
      {
        id: "TF-10",
        statement:
          "The E-Commerce/GPSR text says online marketplaces do not need recall information and do not need to register on EU Safety Gate.",
        label: "FALSE",
        source: "E-Commerce (GPSR summary)",
      },
    ] as TrueFalseQuestion[],
    scenarioQuestion: {
      id: "SC-01",
      title: "HeatCircle startup scenario in Germany",
      prompt:
        "Two founders in Germany plan a B2C startup named HeatCircle: an online marketplace for smart heating accessories with EUR 12,000 capital. They want limited liability, launch online sales in 10 weeks, and prepare for an angel round in 8 months. Provide legal form recommendation, registration/tax checklist, e-commerce compliance essentials, and an 8-week market validation plan with citations.",
    },
    dsrPhases: [
      "Problem Identification",
      "Objective Definition",
      "Design and Development",
      "Demonstration",
      "Evaluation",
      "Communication",
    ],
  }),
});
