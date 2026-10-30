<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { chatApi } from '@/api'
import { useAuthStore } from '@/stores/auth'
import type { ChatStats, ConversationItem } from '@/types'

// 个人中心：提问统计 + 风险等级 + 会话历史
const auth = useAuthStore()
const stats = ref<ChatStats | null>(null)
const conversations = ref<ConversationItem[]>([])

onMounted(async () => {
  try {
    await auth.fetchMe()
    const [statsResult, conversationsResult] = await Promise.all([
      chatApi.stats(),
      chatApi.listConversations({ page: 1, size: 20 }),
    ])
    stats.value = statsResult
    conversations.value = conversationsResult.items
  } catch (error) {
    ElMessage.error((error as Error).message)
  }
})
</script>

<template>
  <div class="page">
    <h3 class="page-title">个人中心</h3>

    <el-row :gutter="16">
      <el-col :span="6"><el-card shadow="never"><div class="stat-num">{{ stats?.question_count ?? 0 }}</div><div class="stat-label">累计提问</div></el-card></el-col>
      <el-col :span="6"><el-card shadow="never"><div class="stat-num">{{ stats?.today_question_count ?? 0 }}</div><div class="stat-label">今日提问</div></el-card></el-col>
      <el-col :span="6"><el-card shadow="never"><div class="stat-num">{{ stats?.blocked_count ?? 0 }}</div><div class="stat-label">被合规拦截</div></el-card></el-col>
      <el-col :span="6">
        <el-card shadow="never">
          <div class="stat-num">{{ stats?.avg_latency_ms ?? '-' }}</div>
          <div class="stat-label">平均响应耗时(ms)</div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px">
      <el-col :span="8">
        <el-card shadow="never">
          <template #header>我的档案</template>
          <el-descriptions :column="1" size="small" border>
            <el-descriptions-item label="用户名">{{ auth.user?.username }}</el-descriptions-item>
            <el-descriptions-item label="角色">
              {{ auth.isAdmin ? '管理员' : '学生' }}
            </el-descriptions-item>
            <el-descriptions-item label="风险等级">
              {{ auth.user?.risk_level || '未测评' }}
            </el-descriptions-item>
            <el-descriptions-item label="反馈">
              👍 {{ stats?.feedback_up ?? 0 }}　👎 {{ stats?.feedback_down ?? 0 }}
            </el-descriptions-item>
          </el-descriptions>
          <el-button style="margin-top: 12px" @click="$router.push('/risk')">去做风险测评</el-button>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card shadow="never">
          <template #header>我的会话（最近 20 条）</template>
          <el-table :data="conversations" size="small" empty-text="还没有会话记录">
            <el-table-column prop="title" label="标题" min-width="180" show-overflow-tooltip />
            <el-table-column prop="message_count" label="消息数" width="90" />
            <el-table-column label="最近活跃" width="180">
              <template #default="{ row }">{{ row.last_message_at || row.created_at || '-' }}</template>
            </el-table-column>
            <el-table-column label="操作" width="90">
              <template #default>
                <el-button size="small" text type="primary" @click="$router.push('/chat')">继续对话</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.stat-num {
  font-size: 26px;
  font-weight: 600;
  color: #2f6fed;
}

.stat-label {
  margin-top: 4px;
  font-size: 12px;
  color: #909399;
}
</style>
