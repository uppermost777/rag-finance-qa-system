import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import 'element-plus/dist/index.css'
import './styles/global.css'

// 应用入口：Pinia 状态 + 路由
createApp(App).use(createPinia()).use(router).mount('#app')
