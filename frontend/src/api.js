const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

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

  return url.toString();
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
    throw new Error(detail);
  }

  return data;
}

export const api = {
  health: () => request('/api/health'),
  login: (body) => request('/api/auth/login', { method: 'POST', body }),
  getAuthStatus: (email) => request('/api/auth/status', { params: { email } }),
  updateProfile: (body) => request('/api/auth/profile', { method: 'POST', body }),
  registerStart: (body) => request('/api/auth/register/start', { method: 'POST', body }),
  registerVerify: (body) => request('/api/auth/register/verify', { method: 'POST', body }),
  resetPassword: (body) => request('/api/auth/reset-password', { method: 'POST', body }),
  changePassword: (body) => request('/api/auth/change-password', { method: 'POST', body }),
  setupTotp: (body) => request('/api/auth/totp/setup', { method: 'POST', body }),
  enableTotp: (body) => request('/api/auth/totp/enable', { method: 'POST', body }),
  verifyTotp: (body) => request('/api/auth/totp/verify', { method: 'POST', body }),
  disableTotp: (body) => request('/api/auth/totp/disable', { method: 'POST', body }),
  listChats: (user) => request('/api/chats', { params: { user } }),
  getChat: (chatId, user) => request(`/api/chats/${chatId}`, { params: { user } }),
  sendMessage: (chatId, body) => request(`/api/chats/${chatId}/message`, { method: 'POST', body }),
  regenerateResponse: (chatId, user) => request(`/api/chats/${chatId}/regenerate`, { method: 'POST', params: { user } }),
  exportChatPdf: (chatId, user) => request(`/api/chats/${chatId}/export`, { params: { user }, parseAs: 'blob' }),
  sendFeedback: (body) => request('/api/feedback', { method: 'POST', body }),
  getPromptAnalytics: (email) => request('/api/analytics/prompts', { params: { email } }),
  listDocuments: (email) => request('/api/admin/documents', { params: { email } }),
  getAdminStatus: (email) => request('/api/admin/status', { params: { email } }),
  uploadDocuments: (email, formData) => request('/api/admin/upload', { method: 'POST', params: { email }, body: formData }),
  reindexDocuments: (email) => request('/api/admin/reindex', { method: 'POST', params: { email } }),
};
