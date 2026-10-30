<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'

// 登录 / 注册页
const auth = useAuthStore()
const router = useRouter()
const route = useRoute()
const mode = ref<'login' | 'register'>('login')
const loading = ref(false)
const form = reactive({ username: '', password: '', email: '' })

async function submit(): Promise<void> {
  if (!form.username || !form.password) {
    ElMessage.warning('请输入用户名与密码')
    return
  }
  loading.value = true
  try {
    if (mode.value === 'login') {
      await auth.login(form.username, form.password)
      ElMessage.success('登录成功')
    } else {
      await auth.register(form.username, form.password, form.email || undefined)
      ElMessage.success('注册成功，已自动登录')
    }
    const redirect = (route.query.redirect as string) || '/chat'
    await router.push(redirect)
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    loading.value = false
  }
}

function fillDemo(username: string, password: string): void {
  form.username = username
  form.password = password
}
</script>

<template>
  <div class="login-wrap">
    <el-card class="login-card">
      <h2 class="login-title">大学生理财知识问答系统</h2>
      <p class="login-sub">基于 LangChain 与向量数据库的 RAG 问答 · 仅作知识科普</p>

      <el-form label-position="top" @submit.prevent="submit">
        <el-form-item label="用户名">
          <el-input v-model="form.username" placeholder="3~32 位中文、字母、数字或下划线" clearable />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.password" type="password" placeholder="至少 6 位，不要纯数字" show-password />
        </el-form-item>
        <el-form-item v-if="mode === 'register'" label="邮箱（可选）">
          <el-input v-model="form.email" placeholder="用于找回账号" clearable />
        </el-form-item>
      </el-form>

      <el-button type="primary" style="width: 100%" :loading="loading" @click="submit">
        {{ mode === 'login' ? '登录' : '注册并登录' }}
      </el-button>

      <div class="login-switch">
        <el-link type="primary" :underline="false" @click="mode = mode === 'login' ? 'register' : 'login'">
          {{ mode === 'login' ? '没有账号？去注册' : '已有账号？去登录' }}
        </el-link>
      </div>

      <el-divider>演示账号（点击填充）</el-divider>
      <div class="login-demo">
        <el-button size="small" @click="fillDemo('student_lin', 'Student@2026')">学生 student_lin</el-button>
        <el-button size="small" @click="fillDemo('admin', 'Admin@2026')">管理员 admin</el-button>
      </div>
      <div class="disclaimer">
        本系统为本科毕业设计作品，仅用于理财知识科普，不构成任何投资建议；理财有风险，投资需谨慎。
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.login-wrap {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #eef3ff 0%, #f7f9fc 60%, #eef7f1 100%);
}

.login-card {
  width: 420px;
  padding: 8px 12px;
}

.login-title {
  margin: 8px 0 4px;
  font-size: 20px;
  text-align: center;
}

.login-sub {
  margin: 0 0 18px;
  color: #909399;
  font-size: 12px;
  text-align: center;
}

.login-switch {
  margin-top: 10px;
  text-align: center;
}

.login-demo {
  display: flex;
  gap: 8px;
  justify-content: center;
}
</style>
