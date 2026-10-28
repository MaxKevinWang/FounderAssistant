export interface SensitiveInputFinding {
  category: string;
  label: string;
}

const DETECTORS: Array<{ category: string; label: string; pattern: RegExp }> = [
  {
    category: "contact_details",
    label: "email address",
    pattern: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i,
  },
  {
    category: "financial_identifier",
    label: "IBAN",
    pattern: /\b[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]){11,30}\b/i,
  },
  {
    category: "financial_identifier",
    label: "payment-card number",
    pattern: /\b(?:\d[ -]*?){13,19}\b/,
  },
  {
    category: "personal_identifier",
    label: "tax or VAT identifier",
    pattern: /\b(?:VAT|USt-?IdNr\.?|Steuer-?ID|tax\s+ID)\s*[:#-]?\s*[A-Z0-9/ -]{8,20}\b/i,
  },
  {
    category: "credentials",
    label: "password, API key, or access token",
    pattern: /\b(?:password|passwort|api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*\S{6,}/i,
  },
  {
    category: "contact_details",
    label: "phone number",
    pattern: /(?:\+\d{1,3}[\s().-]*)?(?:\d[\s().-]*){9,14}/,
  },
];

export function detectStructuredSensitiveInput(text: string): SensitiveInputFinding[] {
  const findings: SensitiveInputFinding[] = [];
  const seen = new Set<string>();
  for (const detector of DETECTORS) {
    if (!detector.pattern.test(text) || seen.has(detector.label)) continue;
    findings.push({ category: detector.category, label: detector.label });
    seen.add(detector.label);
  }
  return findings;
}
