import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAuth } from '../context/AuthContext.jsx'
import { fetchChats, fetchChatHistory, sendMessage } from '../api/api.js'

const CHAT_ID = 'default'

function BotIcon() {
  return (
    <div className="w-7 h-7 rounded-full bg-amber-500 flex-shrink-0 flex items-center justify-center shadow-[0_0_10px_rgba(245,158,11,0.5)]">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#0d0c0a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7H3a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/>
        <path d="M5 14v7M19 14v7M9 14v7M15 14v7"/>
      </svg>
    </div>
  )
}

export default function Chat() {
  const { email, logout } = useAuth()
  const navigate = useNavigate()

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [chatIds, setChatIds] = useState([])
  const [activeChatId, setActiveChatId] = useState(CHAT_ID)
  const [historyLoading, setHistoryLoading] = useState(true)

  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  // Guard: redirect if not logged in
  useEffect(() => {
    const stored = localStorage.getItem('userEmail')
    if (!stored) navigate('/login')
  }, [navigate])

  // Load sidebar chat list
  useEffect(() => {
    if (!email) return
    fetchChats(email)
      .then((res) => {
        const ids = res.data.chat_ids
        if (!ids.includes(CHAT_ID)) ids.unshift(CHAT_ID)
        setChatIds(ids)
      })
      .catch(() => setChatIds([CHAT_ID]))
  }, [email])

  // Load chat history when active chat changes
  useEffect(() => {
    if (!email) return
    setHistoryLoading(true)
    fetchChatHistory(activeChatId, email)
      .then((res) => setMessages(res.data.messages || []))
      .catch(() => setMessages([]))
      .finally(() => setHistoryLoading(false))
  }, [activeChatId, email])

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSend(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return
    setInput('')

    // Optimistic user bubble
    const optimistic = { role: 'user', content: text, timestamp: new Date().toISOString() }
    setMessages((prev) => [...prev, optimistic])
    setLoading(true)

    try {
      const res = await sendMessage(activeChatId, email, text)
      setMessages(res.data.messages || [])
      // Refresh chat list in sidebar
      fetchChats(email).then((r) => {
        const ids = r.data.chat_ids
        if (!ids.includes(CHAT_ID)) ids.unshift(CHAT_ID)
        setChatIds(ids)
      })
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error: could not reach the server.', timestamp: new Date().toISOString() },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function handleLogout() {
    logout()
    navigate('/login')
  }

  return (
    <div className="h-screen flex bg-ink-950 text-ink-100">
      {/* ── Sidebar ── */}
      <aside className="w-56 flex-shrink-0 flex flex-col bg-ink-900 border-r border-ink-800">
        {/* Brand */}
        <div className="px-5 py-5 border-b border-ink-800">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_8px_#fbbf24]" />
            <span className="font-mono text-xs tracking-[0.25em] uppercase text-ink-300">Chatbot</span>
          </div>
        </div>

        {/* Chats */}
        <div className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          <p className="px-2 mb-2 text-[10px] font-mono tracking-widest uppercase text-ink-500">
            Conversations
          </p>
          {chatIds.map((id) => (
            <button
              key={id}
              onClick={() => setActiveChatId(id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors truncate ${
                activeChatId === id
                  ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                  : 'text-ink-400 hover:bg-ink-800 hover:text-ink-200'
              }`}
            >
              # {id}
            </button>
          ))}
        </div>

        {/* User / logout */}
        <div className="px-4 py-4 border-t border-ink-800">
          <p className="text-xs text-ink-500 truncate mb-2">{email}</p>
          <button
            onClick={handleLogout}
            className="text-xs text-ink-500 hover:text-red-400 transition-colors"
          >
            Sign out →
          </button>
        </div>
      </aside>

      {/* ── Main chat area ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-ink-800 bg-ink-900/60 backdrop-blur">
          <div>
            <h2 className="text-sm font-semibold text-ink-100"># {activeChatId}</h2>
            <p className="text-xs text-ink-500">{messages.length} messages</p>
          </div>
          <span className="flex items-center gap-1.5 text-xs text-green-400">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            Online
          </span>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {historyLoading && (
            <div className="flex justify-center py-8">
              <div className="flex gap-1.5">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-2 h-2 rounded-full bg-amber-500 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          )}

          {!historyLoading && messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-3 py-20">
              <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
                <span className="text-2xl">✦</span>
              </div>
              <p className="text-ink-400 text-sm">Send a message to start the conversation</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex items-start gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              {msg.role === 'assistant' && <BotIcon />}

              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-full flex-shrink-0 bg-ink-700 border border-ink-600 flex items-center justify-center text-xs font-semibold text-ink-300">
                  {email ? email[0].toUpperCase() : 'U'}
                </div>
              )}

              <div
                className={`max-w-[72%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-amber-500/15 border border-amber-500/25 text-ink-100 rounded-tr-sm'
                    : 'bg-ink-800 border border-ink-700 text-ink-200 rounded-tl-sm'
                }`}
              >
                {msg.role === 'assistant' ? (
                  <div className="prose-custom">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-start gap-3">
              <BotIcon />
              <div className="bg-ink-800 border border-ink-700 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1.5">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="px-6 py-4 border-t border-ink-800 bg-ink-900/60 backdrop-blur">
          <form onSubmit={handleSend} className="flex items-end gap-3">
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => {
                setInput(e.target.value)
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSend(e)
                }
              }}
              placeholder="Message the bot… (Enter to send, Shift+Enter for newline)"
              disabled={loading}
              className="flex-1 resize-none bg-ink-800 border border-ink-600 rounded-xl px-4 py-3 text-sm text-ink-100 placeholder-ink-500 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 transition-colors min-h-[46px] max-h-40 overflow-y-auto"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="flex-shrink-0 w-11 h-11 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:bg-ink-700 disabled:cursor-not-allowed flex items-center justify-center transition-all shadow-[0_0_16px_rgba(245,158,11,0.3)] hover:shadow-[0_0_24px_rgba(245,158,11,0.5)]"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0d0c0a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
