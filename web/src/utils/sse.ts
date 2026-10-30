import type { Citation, StreamEventPayload } from '@/types'

export interface StreamHandlers {
  onCitations?: (citations: Citation[]) => void
  onDelta?: (text: string) => void
  onDone?: (payload: StreamEventPayload) => void
  onError?: (message: string) => void
}

export interface StreamOptions extends StreamHandlers {
  question: string
  conversationId?: number | null
  kbIds?: number[]
  strategy?: string
  signal?: AbortSignal
}

function buildUrl(options: StreamOptions): string {
  // 兼容非 Vite 运行环境（如单元测试）：import.meta.env 可能不存在
  const base = import.meta.env?.VITE_API_BASE_URL || '/api/v1'
  const params = new URLSearchParams({ question: options.question })
  if (options.conversationId) params.set('conversation_id', String(options.conversationId))
  if (options.kbIds && options.kbIds.length) params.set('kb_ids', options.kbIds.join(','))
  if (options.strategy) params.set('strategy', options.strategy)
  return `${base}/chat/stream?${params.toString()}`
}

/**
 * 解析 SSE 流：使用 fetch + ReadableStream（相比 EventSource 支持鉴权头与中断控制）。
 * 支持的事件：citations / delta / done / error。
 */
export async function streamAnswer(options: StreamOptions): Promise<void> {
  const token = readToken()
  const response = await fetch(buildUrl(options), {
    method: 'GET',
    headers: {
      Accept: 'text/event-stream',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    signal: options.signal,
  })

  if (!response.ok || !response.body) {
    options.onError?.(`请求失败：HTTP ${response.status}`)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let eventName = 'message'
  const dataLines: string[] = []

  const flush = (): void => {
    if (!dataLines.length) {
      eventName = 'message'
      return
    }
    const raw = dataLines.join('\n')
    dataLines.length = 0
    let payload: StreamEventPayload = {}
    try {
      payload = JSON.parse(raw) as StreamEventPayload
    } catch {
      payload = { text: raw }
    }
    switch (eventName) {
      case 'citations':
        options.onCitations?.(payload.citations || [])
        break
      case 'delta':
        options.onDelta?.(payload.text || '')
        break
      case 'done':
        options.onDone?.(payload)
        break
      case 'error':
        options.onError?.(payload.message || '服务端错误')
        break
      default:
        break
    }
    eventName = 'message'
  }

  // 逐块解析：按行处理，遇到空行表示一个 SSE 事件结束
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let index = buffer.indexOf('\n')
    while (index >= 0) {
      const line = buffer.slice(0, index).replace(/\r$/, '')
      buffer = buffer.slice(index + 1)
      if (line === '') {
        flush()
      } else if (line.startsWith('event:')) {
        eventName = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim())
      }
      index = buffer.indexOf('\n')
    }
  }
  flush()
}

/** 安全读取本地 token（非浏览器环境如单元测试下返回空串）。 */
function readToken(): string {
  try {
    return typeof localStorage === 'undefined' ? '' : localStorage.getItem('finance_qa_token') || ''
  } catch {
    return ''
  }
}
