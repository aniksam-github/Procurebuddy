import { BASE_URL } from './config/api';

const API_BASE = BASE_URL.replace(/\/$/, '');
const SESSION_KEY = 'procurebuddy-session';
const TOKEN_KEYS = ['token', 'accessToken', 'access_token', 'authToken', 'procurebuddy-token'];

function hasAuthorizationHeader(headers) {
  return Object.keys(headers).some((key) => key.toLowerCase() === 'authorization');
}

function getStoredToken() {
  if (typeof window === 'undefined') return '';

  try {
    const session = JSON.parse(window.localStorage.getItem(SESSION_KEY) || 'null');
    const sessionToken = session?.token || session?.accessToken || session?.access_token;
    if (typeof sessionToken === 'string' && sessionToken.trim()) {
      return sessionToken.trim();
    }
  } catch {
    // Fall through to legacy keys.
  }

  const preferredToken = window.localStorage.getItem('procurebuddy-token');
  if (typeof preferredToken === 'string' && preferredToken.trim()) {
    return preferredToken.trim();
  }

  for (const key of TOKEN_KEYS) {
    if (key === 'procurebuddy-token') continue;
    const value = window.localStorage.getItem(key);
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }

  return '';
}

function buildUrl(path, params) {
  const base = API_BASE ? `${API_BASE}${path}` : path;
  const url = new URL(base, window.location.origin);

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, value);
      }
    });
  }

  if (API_BASE) {
    return url.toString();
  }

  return `${url.pathname}${url.search}${url.hash}`;
}

async function request(path, options = {}) {
  const {
    method = 'GET',
    body,
    params,
    headers = {},
    parseAs = 'auto',
  } = options;

  const requestHeaders = { ...headers };
  const token = getStoredToken();
  const requestHasAuthorization = hasAuthorizationHeader(requestHeaders);

  if (token && !requestHasAuthorization) {
    requestHeaders.Authorization = `Bearer ${token}`;
  }

  if (import.meta.env.DEV && path.startsWith('/api/chats')) {
    console.debug('[api request]', method, path, {
      hasAuthorization: Boolean(requestHeaders.Authorization),
      authorizationPrefix: requestHeaders.Authorization
        ? `${String(requestHeaders.Authorization).slice(0, 16)}...`
        : '',
    });
  }

  let payload = body;

  if (body && !(body instanceof FormData)) {
    requestHeaders['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  const response = await fetch(buildUrl(path, params), {
    method,
    headers: requestHeaders,
    body: payload,
  });

  const contentType = response.headers.get('content-type') || '';
  let data;
  if (parseAs === 'blob') {
    data = await response.blob();
  } else if (parseAs === 'text') {
    data = await response.text();
  } else {
    data = contentType.includes('application/json')
      ? await response.json()
      : await response.text();
  }

  if (!response.ok) {
    const detail =
      typeof data === 'string'
        ? data
        : data?.detail || data?.message || 'Request failed.';
    const error = new Error(detail);
    error.status = response.status;
    error.payload = data;
    throw error;
  }

  return data;
}

export const api = {
  health: () => request('/api/health'),
  login: (body) => request('/api/auth/login', { method: 'POST', body }),
  getAuthStatus: (email, options = {}) => request('/api/auth/status', { params: { email }, ...options }),
  updateProfile: (body) => request('/api/auth/profile', { method: 'POST', body }),
  registerStart: (body) => request('/api/auth/register/start', { method: 'POST', body }),
  registerVerify: (body) => request('/api/auth/register/verify', { method: 'POST', body }),
  resetPassword: (body) => request('/api/auth/reset-password', { method: 'POST', body }),
  changePassword: (body) => request('/api/auth/change-password', { method: 'POST', body }),
  setupTotp: (body) => request('/api/auth/totp/setup', { method: 'POST', body }),
  enableTotp: (body) => request('/api/auth/totp/enable', { method: 'POST', body }),
  verifyTotp: (body) => request('/api/auth/totp/verify', { method: 'POST', body }),
  disableTotp: (body) => request('/api/auth/totp/disable', { method: 'POST', body }),
  listChats: (user, options = {}) => request('/api/chats', { params: { user }, ...options }),
  getChat: (chatId, user, options = {}) => request(`/api/chats/${chatId}`, { params: { user }, ...options }),
  sendMessage: (chatId, body, options = {}) => request(`/api/chats/${chatId}/message`, { method: 'POST', body, ...options }),
  regenerateResponse: (chatId, user, options = {}) => request(`/api/chats/${chatId}/regenerate`, { method: 'POST', params: { user }, ...options }),
  exportChatPdf: (chatId, user, options = {}) => request(`/api/chats/${chatId}/export`, { params: { user }, parseAs: 'blob', ...options }),
  sendFeedback: (body) => request('/api/feedback', { method: 'POST', body }),
  getPromptAnalytics: (email) => request('/api/analytics/prompts', { params: { email } }),
  listDocuments: (email) => request('/api/admin/documents', { params: { email } }),
  getAdminStatus: (email) => request('/api/admin/status', { params: { email } }),
  uploadDocuments: (email, formData) => request('/api/admin/upload', { method: 'POST', params: { email }, body: formData }),
  reindexDocuments: (email) => request('/api/admin/reindex', { method: 'POST', params: { email } }),
};
