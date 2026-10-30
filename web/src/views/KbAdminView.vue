<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { docApi, kbApi } from '@/api'
import type { DocumentItem, KbItem } from '@/types'

// 知识库与文档管理（管理员）：上传、进度、重建索引、删除
const kbs = ref<KbItem[]>([])
const currentKbId = ref<number | null>(null)
const documents = ref<DocumentItem[]>([])
const loading = ref(false)
const uploadRef = ref()
const dialogVisible = ref(false)
const newKb = ref({ name: '', category: '基金投资', description: '' })
let timer: number | null = null

async function loadKbs(): Promise<void> {
  loading.value = true
  try {
    const result = await kbApi.list({ page: 1, size: 100 })
    kbs.value = result.items
    if (!currentKbId.value && result.items.length) {
      currentKbId.value = result.items[0].id
      await loadDocuments()
    }
  } finally {
    loading.value = false
  }
}

async function loadDocuments(): Promise<void> {
  if (!currentKbId.value) return
  const result = await docApi.list({ kb_id: currentKbId.value, page: 1, size: 50 })
  documents.value = result.items
  schedulePolling()
}

function schedulePolling(): void {
  if (timer) window.clearTimeout(timer)
  const pending = documents.value.filter((item) => !['done', 'failed'].includes(item.status))
  if (!pending.length) return
  timer = window.setTimeout(async () => {
    await Promise.all(
      pending.map(async (item) => {
        const info = await docApi.progress(item.id)
        item.status = info.status
        item.progress = info.progress
        item.chunk_count = info.chunk_count
        item.error_msg = info.error_msg
      }),
    )
    schedulePolling()
  }, 2000)
}

async function createKb(): Promise<void> {
  if (!newKb.value.name) {
    ElMessage.warning('请填写知识库名称')
    return
  }
  await kbApi.create({ ...newKb.value })
  dialogVisible.value = false
  newKb.value = { name: '', category: '基金投资', description: '' }
  ElMessage.success('知识库已创建')
  await loadKbs()
}

async function removeKb(kb: KbItem): Promise<void> {
  await ElMessageBox.confirm(`确认删除知识库「${kb.name}」及其全部文档与向量？`, '危险操作', { type: 'warning' })
  await kbApi.remove(kb.id)
  ElMessage.success('已删除')
  currentKbId.value = null
  documents.value = []
  await loadKbs()
}

async function handleUpload(options: { file: File }): Promise<void> {
  if (!currentKbId.value) return
  try {
    const result = await docApi.upload(currentKbId.value, options.file)
    ElMessage.success(result.message)
    await loadDocuments()
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    uploadRef.value?.clearFiles?.()
  }
}

async function reindexDoc(doc: DocumentItem): Promise<void> {
  await docApi.reindex(doc.id)
  ElMessage.success('已提交重建任务')
  await loadDocuments()
}

async function reindexKb(): Promise<void> {
  if (!currentKbId.value) return
  await docApi.reindexKb(currentKbId.value)
  ElMessage.success('已提交整库重建任务')
  await loadDocuments()
}

async function removeDoc(doc: DocumentItem): Promise<void> {
  await ElMessageBox.confirm(`确认删除文档「${doc.name}」？向量与切片会一并清理。`, '提示', { type: 'warning' })
  await docApi.remove(doc.id)
  ElMessage.success('已删除')
  await loadDocuments()
}

function statusTag(status: string): 'success' | 'info' | 'warning' | 'danger' {
  if (status === 'done') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'pending') return 'info'
  return 'warning'
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

onMounted(loadKbs)
onUnmounted(() => {
  if (timer) window.clearTimeout(timer)
})
</script>

<template>
  <div class="page">
    <h3 class="page-title">知识库管理（管理员）</h3>

    <el-row :gutter="16">
      <el-col :span="8">
        <el-card shadow="never">
          <template #header>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <span>知识库</span>
              <el-button size="small" type="primary" @click="dialogVisible = true">新建</el-button>
            </div>
          </template>
          <div
            v-for="kb in kbs"
            :key="kb.id"
            class="kb-item"
            :class="{ active: kb.id === currentKbId }"
            @click="((currentKbId = kb.id), loadDocuments())"
          >
            <div class="kb-name">{{ kb.name }}</div>
            <div class="kb-meta">
              <el-tag size="small">{{ kb.category }}</el-tag>
              <span>{{ kb.doc_count }} 篇 / {{ kb.chunk_count }} 切片</span>
              <el-link type="danger" :underline="false" @click.stop="removeKb(kb)">删除</el-link>
            </div>
          </div>
          <el-empty v-if="!kbs.length" description="暂无知识库" :image-size="60" />
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card shadow="never">
          <template #header>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <span>文档（{{ documents.length }}）</span>
              <div style="display: flex; gap: 8px">
                <el-upload
                  ref="uploadRef"
                  :auto-upload="false"
                  :show-file-list="false"
                  accept=".pdf,.docx,.txt,.md,.html"
                  :on-change="(file: any) => handleUpload({ file: file.raw })"
                >
                  <el-button type="primary" :disabled="!currentKbId">上传文档（≤20MB）</el-button>
                </el-upload>
                <el-button :disabled="!currentKbId" @click="reindexKb">整库重建索引</el-button>
              </div>
            </div>
          </template>

          <el-table :data="documents" v-loading="loading" size="small" empty-text="暂无文档，请先上传">
            <el-table-column prop="name" label="文件名" min-width="180" show-overflow-tooltip />
            <el-table-column prop="file_type" label="类型" width="70" />
            <el-table-column label="大小" width="90">
              <template #default="{ row }">{{ formatSize(row.file_size) }}</template>
            </el-table-column>
            <el-table-column label="状态" width="150">
              <template #default="{ row }">
                <el-tag size="small" :type="statusTag(row.status)">{{ row.status }}</el-tag>
                <el-progress
                  v-if="!['done', 'failed'].includes(row.status)"
                  :percentage="row.progress"
                  :stroke-width="6"
                  style="margin-top: 4px"
                />
                <el-tooltip v-if="row.error_msg" :content="row.error_msg">
                  <el-tag size="small" type="danger" style="margin-left: 4px">查看原因</el-tag>
                </el-tooltip>
              </template>
            </el-table-column>
            <el-table-column prop="chunk_count" label="切片" width="70" />
            <el-table-column label="操作" width="150">
              <template #default="{ row }">
                <el-button size="small" text type="primary" @click="reindexDoc(row)">重建</el-button>
                <el-button size="small" text type="danger" @click="removeDoc(row)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>

    <el-dialog v-model="dialogVisible" title="新建知识库" width="420px">
      <el-form label-width="80px">
        <el-form-item label="名称"><el-input v-model="newKb.name" placeholder="如：银行储蓄与存款保险" /></el-form-item>
        <el-form-item label="分类">
          <el-select v-model="newKb.category" style="width: 100%">
            <el-option v-for="item in ['储蓄存款', '基金投资', '股票基础', '保险保障', '信用借贷', '反诈防骗']" :key="item" :label="item" :value="item" />
          </el-select>
        </el-form-item>
        <el-form-item label="描述"><el-input v-model="newKb.description" type="textarea" :rows="3" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="createKb">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.kb-item {
  padding: 10px;
  border-radius: 6px;
  border: 1px solid transparent;
  cursor: pointer;
  margin-bottom: 8px;
}

.kb-item:hover {
  background: #f5f7fb;
}

.kb-item.active {
  background: #ecf3ff;
  border-color: #c6dbff;
}

.kb-name {
  font-size: 14px;
  margin-bottom: 4px;
}

.kb-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #909399;
}
</style>
