import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios'

// 统一 Axios 实例：注入 JWT、解包 {code,message,data}、集中错误提示
// 兼容非 Vite 运行环境：import.meta.env 可能不存在
const baseURL = import.meta.env?.VITE_API_BASE_URL || '/api/v1'

export function getToken(): string {
  return localStorage.getItem('finance_qa_token') || ''
}

export function setToken(token: string): void {
  localStorage.setItem('finance_qa_token', token)
}

export function clearToken(): void {
  localStorage.removeItem('finance_qa_token')
}

const http: AxiosInstance = axios.create({
  baseURL,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

http.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers = config.headers ?? {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

http.interceptors.response.use(
  (response) => {
    const body = response.data
    if (body && typeof body === 'object' && 'code' in body) {
      if (body.code === 0) {
        return body.data
      }
      return Promise.reject(new Error(body.message || '请求失败'))
    }
    return body
  },
  (error) => {
    const status = error?.response?.status
    const message = error?.response?.data?.message || error.message || '网络异常，请稍后重试'
    if (status === 401) {
      clearToken()
      if (!location.pathname.startsWith('/login')) {
        location.href = '/login'
      }
    }
    return Promise.reject(new Error(message))
  },
)

export async function request<T>(config: AxiosRequestConfig): Promise<T> {
  return http.request(config) as unknown as Promise<T>
}

export default http
