import DOMPurify from "dompurify";
import { Marked } from "marked";

const marked = new Marked({
  breaks: true,
  gfm: true,
});

export function renderMarkdown(source: string): string {
  const html = marked.parse(source ?? "", { async: false }) as string;

  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
  });
}
