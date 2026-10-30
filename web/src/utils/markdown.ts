import DOMPurify from 'dompurify'
import hljs from 'highlight.js'
import { Marked } from 'marked'

const marked = new Marked({
  gfm: true,
  breaks: true,
})

/**
 * 渲染 Markdown：代码块高亮 + XSS 过滤（模型输出不可信，必须消毒后再插入 DOM）。
 */
export function renderMarkdown(text: string): string {
  const html = marked.parse(text || '') as string
  return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } })
}

/** 对页面中已插入的代码块做高亮。 */
export function highlightAll(root: HTMLElement | null): void {
  if (!root) return
  root.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightElement(block as HTMLElement)
  })
}
