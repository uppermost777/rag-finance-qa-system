// 与后端 Pydantic 模型对应的前端类型定义

export interface ApiResponse<T> {
  code: number
  message: string
  data: T
}

export interface UserInfo {
  id: number
  username: string
  email?: string | null
  role: string
  risk_level?: string | null
  created_at?: string | null
}

export interface LoginResult {
  access_token: string
  token_type: string
  expires_in: number
  user: UserInfo
}

export interface ConversationItem {
  id: number
  title: string
  kb_ids: number[]
  message_count: number
  last_message_at?: string | null
  created_at?: string | null
}

export interface Citation {
  index: number
  chunk_id?: number
  doc_id?: number
  kb_id?: number
  doc_name?: string
  page_no?: number
  snippet?: string
  score?: number
}

export interface MessageItem {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  latency_ms?: number | null
  blocked: boolean
  feedback?: number | null
  created_at?: string | null
}

export interface ChatResult {
  conversation_id: number
  message_id: number
  user_message_id: number
  qa_log_id?: number | null
  answer: string
  citations: Citation[]
  blocked: boolean
  guard_action: string
  strategy: string
  need_human_hint: boolean
  latency_ms: number
}

export interface KbItem {
  id: number
  name: string
  category: string
  description?: string | null
  doc_count: number
  chunk_count: number
  status: number
  created_at?: string | null
}

export interface DocumentItem {
  id: number
  kb_id: number
  name: string
  file_type: string
  file_size: number
  status: string
  progress: number
  page_count: number
  chunk_count: number
  error_msg?: string | null
  created_at?: string | null
}

export interface ProgressInfo {
  doc_id: number
  status: string
  progress: number
  chunk_count: number
  message?: string | null
  error_msg?: string | null
}

export interface RiskDimension {
  key: string
  label: string
  percent: number
}

export interface RiskProfile {
  risk_level: string
  risk_label?: string
  risk_score: { total?: number; max_total?: number; dimensions?: RiskDimension[] }
  suggestion?: string
  learning_path?: string[]
}

export interface RiskQuestion {
  id: number
  dimension: string
  text: string
  options: string[]
}

export interface ChatStats {
  question_count: number
  blocked_count: number
  conversation_count: number
  today_question_count: number
  feedback_up: number
  feedback_down: number
  avg_latency_ms: number | null
  risk_level?: string | null
}

export type StrategyName = 'vector' | 'bm25' | 'hybrid' | 'hybrid_rerank'

export interface StreamEventPayload {
  text?: string
  citations?: Citation[]
  answer?: string
  message_id?: number
  conversation_id?: number
  blocked?: boolean
  guard_action?: string
  need_human_hint?: boolean
  latency_ms?: number
  message?: string
}
