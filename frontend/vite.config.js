import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', '')
  const normalizeTarget = (value) => value ? value.replace(/\/+$/, '') : ''
  const apiProxyTarget = normalizeTarget(env.VITE_API_PROXY_TARGET || 'http://localhost:2002')
  const aiProxyTarget = normalizeTarget(env.VITE_AI_PROXY_TARGET || 'http://localhost:8000')
  const proxy = {}

  if (apiProxyTarget) {
    proxy['/api'] = {
      target: apiProxyTarget,
      changeOrigin: true,
    }
  }

  if (aiProxyTarget) {
    proxy['/ai'] = {
      target: aiProxyTarget,
      changeOrigin: true,
    }
  }

  return {
    envDir: '..',
    plugins: [react()],
    server: {
      port: 5173,
      proxy,
    },
  }
})
