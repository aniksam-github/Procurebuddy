import axios from 'axios'

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api').trim()

const api = axios.create({
  baseURL: apiBaseUrl,
  headers: { 'Content-Type': 'application/json' },
})

// ── Auth ──────────────────────────────────────────────────────────────────────

export const registerStart   = (email)                    => api.post('/auth/register/start',  { email })
export const registerVerify  = (email, otp, password)     => api.post('/auth/register/verify', { email, otp, password })
export const login           = (email, password)          => api.post('/auth/login',           { email, password })
export const changePassword  = (email, new_password)      => api.post('/auth/change-password', { email, new_password })

// ── Chats ─────────────────────────────────────────────────────────────────────

export const fetchChats      = (user)                => api.get(`/chats`,             { params: { user } })
export const fetchChatHistory= (chatId, user)        => api.get(`/chats/${chatId}`,   { params: { user } })
export const sendMessage     = (chatId, user, message) => api.post(`/chats/${chatId}/message`, { user, message })

export default api
