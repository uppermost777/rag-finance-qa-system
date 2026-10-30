import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/chat' },
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: { public: true, title: '登录 / 注册' },
  },
  { path: '/chat', name: 'chat', component: () => import('@/views/ChatView.vue'), meta: { title: '智能问答' } },
  { path: '/risk', name: 'risk', component: () => import('@/views/RiskView.vue'), meta: { title: '风险测评' } },
  {
    path: '/kb',
    name: 'kb',
    component: () => import('@/views/KbAdminView.vue'),
    meta: { title: '知识库管理', admin: true },
  },
  {
    path: '/profile',
    name: 'profile',
    component: () => import('@/views/ProfileView.vue'),
    meta: { title: '个人中心' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 路由守卫：未登录跳登录页；管理员页面校验角色
router.beforeEach((to) => {
  const auth = useAuthStore()
  if (to.meta.public) return true
  if (!auth.token) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  if (to.meta.admin && !auth.isAdmin) {
    return { name: 'chat' }
  }
  return true
})

export default router
