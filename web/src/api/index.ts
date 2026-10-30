import { request } from './http'
import type {
  ChatResult,
  ChatStats,
  ConversationItem,
  DocumentItem,
  KbItem,
  LoginResult,
  MessageItem,
  ProgressInfo,
  RiskProfile,
  RiskQuestion,
  UserInfo,
} from '@/types'

// ---------------------------------------------------------------- 认证
export const authApi = {
  register: (payload: { username: string; password: string; email?: string }) =>
    request<UserInfo>({ url: '/auth/register', method: 'post', data: payload }),
  login: (payload: { username: string; password: string }) =>
    request<LoginResult>({ url: '/auth/login', method: 'post', data: payload }),
  me: () => request<UserInfo>({ url: '/auth/me', method: 'get' }),
  riskQuestions: () =>
    request<{ total: number; questions: RiskQuestion[] }>({ url: '/auth/risk-questions', method: 'get' }),
  submitRisk: (answers: number[]) =>
    request<RiskProfile>({ url: '/auth/risk-profile', method: 'put', data: { answers } }),
  getRisk: () => request<RiskProfile & { message?: string }>({ url: '/auth/risk-profile', method: 'get' }),
}

// ---------------------------------------------------------------- 对话
export const chatApi = {
  createConversation: (payload: { title?: string; kb_ids: number[] }) =>
    request<ConversationItem>({ url: '/chat/conversations', method: 'post', data: payload }),
  listConversations: (params: { page: number; size: number }) =>
    request<{ total: number; page: number; size: number; items: ConversationItem[] }>({
      url: '/chat/conversations',
      method: 'get',
      params,
    }),
  updateConversation: (id: number, payload: { title?: string; kb_ids?: number[] }) =>
    request<ConversationItem>({ url: `/chat/conversations/${id}`, method: 'patch', data: payload }),
  deleteConversation: (id: number) =>
    request<{ id: number }>({ url: `/chat/conversations/${id}`, method: 'delete' }),
  listMessages: (id: number, params: { page: number; size: number }) =>
    request<{ total: number; items: MessageItem[] }>({
      url: `/chat/conversations/${id}/messages`,
      method: 'get',
      params,
    }),
  completions: (payload: { question: string; conversation_id?: number; kb_ids?: number[]; strategy?: string }) =>
    request<ChatResult>({ url: '/chat/completions', method: 'post', data: payload }),
  feedback: (messageId: number, feedback: 1 | -1) =>
    request<{ message_id: number; feedback: number }>({
      url: `/chat/messages/${messageId}/feedback`,
      method: 'post',
      data: { feedback },
    }),
  stats: () => request<ChatStats>({ url: '/chat/stats', method: 'get' }),
}

// ---------------------------------------------------------------- 知识库与文档
export const kbApi = {
  list: (params: { page?: number; size?: number; keyword?: string; only_enabled?: boolean } = {}) =>
    request<{ total: number; items: KbItem[] }>({ url: '/kb', method: 'get', params }),
  create: (payload: { name: string; category: string; description?: string }) =>
    request<KbItem>({ url: '/kb', method: 'post', data: payload }),
  update: (id: number, payload: { name?: string; description?: string; category?: string; status?: number }) =>
    request<KbItem>({ url: `/kb/${id}`, method: 'patch', data: payload }),
  remove: (id: number) => request<{ id: number }>({ url: `/kb/${id}`, method: 'delete' }),
}

export const docApi = {
  upload: (kbId: number, file: File) => {
    const form = new FormData()
    form.append('file', file)
    form.append('kb_id', String(kbId))
    return request<{ doc_id: number; task_id?: string; status: string; message: string }>({
      url: '/document/upload',
      method: 'post',
      data: form,
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  list: (params: { kb_id?: number; status?: string; page?: number; size?: number } = {}) =>
    request<{ total: number; items: DocumentItem[] }>({ url: '/document', method: 'get', params }),
  progress: (docId: number) => request<ProgressInfo>({ url: `/document/${docId}/progress`, method: 'get' }),
  reindex: (docId: number) =>
    request<{ doc_id: number; task_id?: string }>({ url: `/document/${docId}/reindex`, method: 'post' }),
  reindexKb: (kbId: number) =>
    request<{ kb_id: number; task_id?: string }>({ url: '/document/reindex', method: 'post', params: { kb_id: kbId } }),
  remove: (docId: number) => request<{ doc_id: number }>({ url: `/document/${docId}`, method: 'delete' }),
}

// ---------------------------------------------------------------- 评估
export const evalApi = {
  run: (payload: { dataset?: string; limit?: number; strategies?: string[] }) =>
    request<{ run_id: string; status: string; message: string }>({ url: '/evaluation/run', method: 'post', data: payload }),
  status: (runId: string) =>
    request<{ run_id: string; status: string; progress: number; report_path?: string; message?: string }>({
      url: `/evaluation/${runId}`,
      method: 'get',
    }),
  report: (runId?: string) =>
    request<{ columns: string[]; rows: Record<string, string | number>[] }>({
      url: '/evaluation/report',
      method: 'get',
      params: runId ? { run_id: runId } : {},
    }),
}
