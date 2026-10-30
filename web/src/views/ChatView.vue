<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { chatApi, kbApi } from '@/api'
import { streamAnswer } from '@/utils/sse'
import { highlightAll, renderMarkdown } from '@/utils/markdown'
import type { ChatStats, Citation, ConversationItem, KbItem, MessageItem, StrategyName } from '@/types'

// ------------------------------------------------------------------ 状态
const conversations = ref<ConversationItem[]>([])
const currentId = ref<number | null>(null)
const messages = ref<MessageItem[]>([])
const kbOptions = ref<KbItem[]>([])
const selectedKbIds = ref<number[]>([])
const strategy = ref<StrategyName>('hybrid_rerank')
const question = ref('')
const streaming = ref(false)
const streamText = ref('')
const streamCitations = ref<Citation[]>([])
const listRef = ref<HTMLElement | null>(null)
const stats = ref<ChatStats | null>(null)
let controller: AbortController | null = null

const currentConversation = computed(() =>
  conversations.value.find((item) => item.id === currentId.value) || null,
)

// ------------------------------------------------------------------ 初始化
async function loadKbs(): Promise<void> {
  const result = await kbApi.list({ page: 1, size: 100, only_enabled: true })
  kbOptions.value = result.items
}

async function loadConversations(): Promise<void> {
  const result = await chatApi.listConversations({ page: 1, size: 50 })
  conversations.value = result.items
  if (!currentId.value && result.items.length) {
    await selectConversation(result.items[0].id)
  }
}

async function loadStats(): Promise<void> {
  stats.value = await chatApi.stats()
}

async function selectConversation(id: number): Promise<void> {
  currentId.value = id
  const conversation = conversations.value.find((item) => item.id === id)
  selectedKbIds.value = conversation?.kb_ids ?? []
  const result = await chatApi.listMessages(id, { page: 1, size: 200 })
  messages.value = result.items
  await scrollToBottom()
}

async function createConversation(): Promise<void> {
  const created = await chatApi.createConversation({ title: '新会话', kb_ids: selectedKbIds.value })
  conversations.value.unshift(created)
  currentId.value = created.id
  messages.value = []
}

async function removeConversation(id: number): Promise<void> {
  await ElMessageBox.confirm('确认删除该会话？历史消息将被隐藏。', '提示', { type: 'warning' })
  await chatApi.deleteConversation(id)
  conversations.value = conversations.value.filter((item) => item.id !== id)
  if (currentId.value === id) {
    currentId.value = null
    messages.value = []
    if (conversations.value.length) await selectConversation(conversations.value[0].id)
  }
}

async function scrollToBottom(): Promise<void> {
  await nextTick()
  if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight
  highlightAll(listRef.value)
}

// ------------------------------------------------------------------ 提问（SSE 流式）
async function send(): Promise<void> {
  const text = question.value.trim()
  if (!text) return
  if (streaming.value) {
    ElMessage.warning('正在回答中，请稍候或点击「停止」')
    return
  }
  if (!currentId.value) await createConversation()

  messages.value.push({ id: -1, role: 'user', content: text, citations: [], blocked: false })
  question.value = ''
  streaming.value = true
  streamText.value = ''
  streamCitations.value = []
  controller = new AbortController()
  await scrollToBottom()

  const conversationId = currentId.value
  try {
    await streamAnswer({
      question: text,
      conversationId,
      kbIds: selectedKbIds.value,
      strategy: strategy.value,
      signal: controller.signal,
      onCitations: (citations) => {
        streamCitations.value = citations
        void scrollToBottom()
      },
      onDelta: (delta) => {
        streamText.value += delta
        void scrollToBottom()
      },
      onError: (message) => ElMessage.error(message),
      onDone: (payload) => {
        messages.value.push({
          id: payload.message_id ?? -1,
          role: 'assistant',
          content: payload.answer ?? streamText.value,
          citations: payload.citations ?? streamCitations.value,
          latency_ms: payload.latency_ms ?? null,
          blocked: Boolean(payload.blocked),
          feedback: null,
        })
        streamText.value = ''
        streamCitations.value = []
        if (payload.need_human_hint) {
          ElMessage.info('知识库中暂无直接相关内容，建议更换问法或上传资料')
        }
        if (conversationId) void loadConversations()
        void loadStats()
      },
    })
  } catch (error) {
    if ((error as Error).name === 'AbortError') {
      ElMessage.info('已停止本次回答')
    } else {
      ElMessage.error((error as Error).message)
    }
  } finally {
    streaming.value = false
    controller = null
    await scrollToBottom()
  }
}

function stop(): void {
  controller?.abort()
  streaming.value = false
}

async function feedback(message: MessageItem, value: 1 | -1): Promise<void> {
  if (message.id < 0) return
  await chatApi.feedback(message.id, value)
  message.feedback = value
  ElMessage.success(value === 1 ? '感谢反馈' : '已记录，我们会持续改进')
}

async function saveKbScope(): Promise<void> {
  if (!currentId.value) return
  await chatApi.updateConversation(currentId.value, { kb_ids: selectedKbIds.value })
  ElMessage.success('检索范围已更新')
}

onMounted(async () => {
  try {
    await Promise.all([loadKbs(), loadConversations(), loadStats()])
  } catch (error) {
    ElMessage.error((error as Error).message)
  }
})
</script>

<template>
  <div class="chat-page">
    <!-- 左侧会话列表 -->
    <aside class="chat-aside">
      <el-button type="primary" style="width: 100%" @click="createConversation">＋ 新建会话</el-button>
      <div class="conv-list">
        <div
          v-for="item in conversations"
          :key="item.id"
          class="conv-item"
          :class="{ active: item.id === currentId }"
          @click="selectConversation(item.id)"
        >
          <div class="conv-title">{{ item.title }}</div>
          <div class="conv-meta">
            <span>{{ item.message_count }} 条消息</span>
            <el-link type="danger" :underline="false" @click.stop="removeConversation(item.id)">删除</el-link>
          </div>
        </div>
        <el-empty v-if="!conversations.length" description="还没有会话，点击上方新建" :image-size="60" />
      </div>
    </aside>

    <!-- 主对话区 -->
    <section class="chat-main">
      <div class="chat-toolbar">
        <el-select
          v-model="selectedKbIds"
          multiple
          collapse-tags
          placeholder="检索范围（默认全部知识库）"
          style="width: 320px"
          @change="saveKbScope"
        >
          <el-option v-for="kb in kbOptions" :key="kb.id" :label="kb.name" :value="kb.id" />
        </el-select>
        <el-select v-model="strategy" style="width: 190px" title="检索策略（对比实验用）">
          <el-option label="混合检索 + 重排" value="hybrid_rerank" />
          <el-option label="混合检索（RRF）" value="hybrid" />
          <el-option label="纯向量检索" value="vector" />
          <el-option label="纯 BM25 检索" value="bm25" />
        </el-select>
        <span v-if="stats" class="chat-stats">
          今日提问 {{ stats.today_question_count }} 次 · 累计 {{ stats.question_count }} 次
          <template v-if="stats.avg_latency_ms"> · 平均 {{ stats.avg_latency_ms }} ms</template>
        </span>
      </div>

      <div ref="listRef" class="chat-list">
        <el-empty
          v-if="!messages.length && !streaming"
          description="试试问：什么是基金定投 / 存款保险保多少 / 校园贷有哪些风险"
        />

        <div v-for="(message, index) in messages" :key="index" class="msg" :class="message.role">
          <div class="bubble">
            <div class="bubble-role">
              {{ message.role === 'user' ? '我' : '理财助手' }}
              <el-tag v-if="message.blocked" size="small" type="danger" style="margin-left: 6px">合规拦截</el-tag>
            </div>
            <div class="markdown-body" v-html="renderMarkdown(message.content)" />

            <el-collapse v-if="message.citations?.length" class="citation-list">
              <el-collapse-item :title="`参考来源（${message.citations.length} 条）`">
                <div v-for="citation in message.citations" :key="citation.index" class="citation-item">
                  <div class="citation-head">
                    <el-tag size="small">[{{ citation.index }}]</el-tag>
                    <span>{{ citation.doc_name || '知识库文档' }}</span>
                    <span v-if="citation.page_no">第 {{ citation.page_no }} 页</span>
                    <span v-if="citation.score !== undefined">相似度 {{ Number(citation.score).toFixed(3) }}</span>
                  </div>
                  <div class="citation-snippet">{{ citation.snippet }}</div>
                </div>
              </el-collapse-item>
            </el-collapse>

            <div v-if="message.role === 'assistant'" class="bubble-actions">
              <el-button
                size="small"
                :type="message.feedback === 1 ? 'primary' : 'default'"
                text
                @click="feedback(message, 1)"
              >
                👍 有帮助
              </el-button>
              <el-button
                size="small"
                :type="message.feedback === -1 ? 'danger' : 'default'"
                text
                @click="feedback(message, -1)"
              >
                👎 需改进
              </el-button>
              <span v-if="message.latency_ms" class="latency">{{ message.latency_ms }} ms</span>
            </div>
          </div>
        </div>

        <!-- 流式输出中 -->
        <div v-if="streaming" class="msg assistant">
          <div class="bubble">
            <div class="bubble-role">理财助手 <el-tag size="small" type="success">生成中…</el-tag></div>
            <el-collapse v-if="streamCitations.length" class="citation-list">
              <el-collapse-item :title="`检索到 ${streamCitations.length} 条参考来源（点击查看）`">
                <div v-for="citation in streamCitations" :key="citation.index" class="citation-item">
                  <div class="citation-head">
                    <el-tag size="small">[{{ citation.index }}]</el-tag>
                    <span>{{ citation.doc_name }}</span>
                    <span v-if="citation.page_no">第 {{ citation.page_no }} 页</span>
                  </div>
                  <div class="citation-snippet">{{ citation.snippet }}</div>
                </div>
              </el-collapse-item>
            </el-collapse>
            <div class="markdown-body" v-html="renderMarkdown(streamText || '检索中…')" />
          </div>
        </div>
      </div>

      <div class="chat-input">
        <el-input
          v-model="question"
          type="textarea"
          :rows="2"
          maxlength="500"
          show-word-limit
          placeholder="输入你的理财问题，例如：每个月剩 500 块怎么安排？（Enter 发送，Shift+Enter 换行）"
          @keydown.enter.exact.prevent="send"
        />
        <div class="chat-input-actions">
          <el-button v-if="streaming" type="danger" plain @click="stop">停止</el-button>
          <el-button type="primary" :loading="streaming" @click="send">发送</el-button>
        </div>
      </div>
      <div class="disclaimer">
        回答基于知识库资料生成，仅供知识科普，不构成投资建议；请通过正规持牌机构办理金融业务。
      </div>
    </section>
  </div>
</template>

<style scoped>
.chat-page {
  display: flex;
  height: calc(100vh - 60px);
  background: #f5f7fb;
}

.chat-aside {
  width: 260px;
  padding: 14px;
  background: #fff;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.conv-list {
  flex: 1;
  overflow-y: auto;
}

.conv-item {
  padding: 10px;
  margin-bottom: 8px;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid transparent;
}

.conv-item:hover {
  background: #f5f7fb;
}

.conv-item.active {
  background: #ecf3ff;
  border-color: #c6dbff;
}

.conv-title {
  font-size: 14px;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conv-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #909399;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 14px 18px;
  min-width: 0;
}

.chat-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.chat-stats {
  color: #909399;
  font-size: 12px;
}

.chat-list {
  flex: 1;
  overflow-y: auto;
  padding-right: 6px;
}

.msg {
  display: flex;
  margin-bottom: 14px;
}

.msg.user {
  justify-content: flex-end;
}

.bubble {
  max-width: 78%;
  padding: 12px 14px;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

.msg.user .bubble {
  background: #2f6fed;
  color: #fff;
}

.bubble-role {
  display: flex;
  align-items: center;
  font-size: 12px;
  color: #909399;
  margin-bottom: 6px;
}

.msg.user .bubble-role {
  color: rgba(255, 255, 255, 0.85);
}

.bubble-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
}

.latency {
  margin-left: auto;
  font-size: 12px;
  color: #909399;
}

.chat-input {
  margin-top: 10px;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.chat-input-actions {
  display: flex;
  gap: 8px;
}
</style>
