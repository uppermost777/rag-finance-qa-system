<script setup lang="ts">
import { onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

// 顶层布局：顶部导航 + 内容区 + 固定免责声明
const auth = useAuthStore()
const route = useRoute()
const router = useRouter()

const menu = [
  { path: '/chat', label: '智能问答' },
  { path: '/risk', label: '风险测评' },
  { path: '/kb', label: '知识库管理', admin: true },
  { path: '/profile', label: '个人中心' },
]

function go(path: string): void {
  router.push(path)
}

function logout(): void {
  auth.logout()
  router.push('/login')
}

onMounted(() => {
  void auth.fetchMe()
})
</script>

<template>
  <div style="min-height: 100%; display: flex; flex-direction: column">
    <el-header
      v-if="route.name !== 'login'"
      style="display: flex; align-items: center; gap: 24px; background: #fff; border-bottom: 1px solid #e4e7ed"
    >
      <span style="font-size: 17px; font-weight: 600; color: #2f6fed">大学生理财知识问答</span>
      <div style="display: flex; gap: 8px; flex: 1">
        <template v-for="item in menu" :key="item.path">
          <el-button
            v-if="!item.admin || auth.isAdmin"
            :type="route.path === item.path ? 'primary' : 'default'"
            link
            @click="go(item.path)"
          >
            {{ item.label }}
          </el-button>
        </template>
      </div>
      <el-dropdown>
        <span style="cursor: pointer">
          {{ auth.displayName }}
          <el-tag v-if="auth.isAdmin" size="small" type="warning" style="margin-left: 6px">管理员</el-tag>
        </span>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item @click="go('/profile')">个人中心</el-dropdown-item>
            <el-dropdown-item divided @click="logout">退出登录</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </el-header>

    <el-main style="padding: 0">
      <router-view />
    </el-main>
  </div>
</template>
