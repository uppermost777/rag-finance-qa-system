import { defineStore } from 'pinia'
import { authApi } from '@/api'
import { clearToken, getToken, setToken } from '@/api/http'
import type { UserInfo } from '@/types'

interface AuthState {
  token: string
  user: UserInfo | null
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    token: getToken(),
    user: null,
  }),
  getters: {
    isAdmin: (state): boolean => state.user?.role === 'admin',
    displayName: (state): string => state.user?.username || '未登录',
  },
  actions: {
    async login(username: string, password: string): Promise<void> {
      const result = await authApi.login({ username, password })
      this.token = result.access_token
      this.user = result.user
      setToken(result.access_token)
    },
    async register(username: string, password: string, email?: string): Promise<void> {
      await authApi.register({ username, password, email })
      await this.login(username, password)
    },
    async fetchMe(): Promise<void> {
      if (!this.token) return
      try {
        this.user = await authApi.me()
      } catch {
        this.logout()
      }
    },
    logout(): void {
      this.token = ''
      this.user = null
      clearToken()
    },
  },
})
