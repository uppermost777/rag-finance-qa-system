<script setup lang="ts">
import { nextTick, onMounted, ref, shallowRef } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import { authApi } from '@/api'
import type { RiskProfile, RiskQuestion } from '@/types'

// 风险测评：10 题问卷 → 五维雷达图 → 等级与学习路径
const questions = ref<RiskQuestion[]>([])
const answers = ref<number[]>([])
const profile = ref<RiskProfile | null>(null)
const submitting = ref(false)
const chartRef = ref<HTMLElement | null>(null)
const chart = shallowRef<echarts.ECharts | null>(null)

async function load(): Promise<void> {
  const result = await authApi.riskQuestions()
  questions.value = result.questions
  answers.value = new Array(result.questions.length).fill(-1)
  try {
    const existing = await authApi.getRisk()
    if (existing.risk_level) {
      profile.value = existing
      await nextTick()
      renderChart(existing)
    }
  } catch {
    /* 尚未测评过，忽略 */
  }
}

async function submit(): Promise<void> {
  if (answers.value.some((item) => item < 0)) {
    ElMessage.warning('请完成全部 10 道题')
    return
  }
  submitting.value = true
  try {
    const result = await authApi.submitRisk(answers.value)
    profile.value = result
    await nextTick()
    renderChart(result)
    ElMessage.success(`测评完成：${result.risk_level} ${result.risk_label ?? ''}`)
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    submitting.value = false
  }
}

function renderChart(result: RiskProfile): void {
  if (!chartRef.value) return
  chart.value = chart.value ?? echarts.init(chartRef.value)
  const dimensions = result.risk_score?.dimensions ?? []
  chart.value.setOption({
    tooltip: {},
    radar: {
      indicator: dimensions.map((item) => ({ name: item.label, max: 100 })),
      radius: '68%',
    },
    series: [
      {
        type: 'radar',
        areaStyle: { opacity: 0.25 },
        data: [{ value: dimensions.map((item) => item.percent), name: '五维得分' }],
      },
    ],
  })
}

onMounted(load)
</script>

<template>
  <div class="page">
    <h3 class="page-title">理财风险测评</h3>
    <p style="color: #909399; font-size: 13px; margin-top: -8px">
      10 道题，约 2 分钟。测评结果用于给出适合你的学习路径，不代表投资能力评价，也不构成投资建议。
    </p>

    <el-row :gutter="16">
      <el-col :span="14">
        <el-card shadow="never">
          <div v-for="(question, index) in questions" :key="question.id" class="question">
            <div class="question-text">{{ index + 1 }}. {{ question.text }}</div>
            <el-radio-group v-model="answers[index]">
              <el-radio v-for="(option, optionIndex) in question.options" :key="option" :value="optionIndex">
                {{ option }}
              </el-radio>
            </el-radio-group>
          </div>
          <el-button type="primary" :loading="submitting" :disabled="!questions.length" @click="submit">
            提交测评
          </el-button>
        </el-card>
      </el-col>

      <el-col :span="10">
        <el-card shadow="never">
          <template #header>测评结果</template>
          <el-empty v-if="!profile" description="提交后展示五维雷达图与学习路径" :image-size="80" />
          <template v-else>
            <div class="result-head">
              <el-tag type="warning" size="large">{{ profile.risk_level }} {{ profile.risk_label }}</el-tag>
              <span>总分 {{ profile.risk_score?.total }} / {{ profile.risk_score?.max_total }}</span>
            </div>
            <div ref="chartRef" style="height: 240px" />
            <p class="suggestion">{{ profile.suggestion }}</p>
            <el-divider>推荐学习路径</el-divider>
            <ol class="path">
              <li v-for="item in profile.learning_path" :key="item">{{ item }}</li>
            </ol>
          </template>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.question {
  padding: 10px 0;
  border-bottom: 1px dashed #ebeef5;
}

.question:last-of-type {
  margin-bottom: 14px;
}

.question-text {
  margin-bottom: 6px;
  font-size: 14px;
}

.result-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 13px;
  color: #606266;
}

.suggestion {
  font-size: 13px;
  color: #606266;
  line-height: 1.7;
}

.path {
  margin: 0;
  padding-left: 18px;
  font-size: 13px;
  color: #606266;
  line-height: 1.9;
}
</style>
