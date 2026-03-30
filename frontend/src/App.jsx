import { useEffect, useState } from 'react';
import LoginPage from './LoginPage';
import Layout from './components/Layout';
import { ChatView, ProfileModal, SettingsModal, SettingsView } from './Views';
import { useTheme } from './context/ThemeContext';
import { useSeasonal } from './context/SeasonalContext';
import { api } from './api';
import './index.css';

const SESSION_KEY = 'procurebuddy-session';
const DRAFT_CHATS_KEY = 'procurebuddy-drafts';
const SIDEBAR_COLLAPSED_KEY = 'procurebuddy-sidebar-collapsed';

function readJson(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function createDraftChat() {
  return {
    chat_id: crypto.randomUUID(),
    title: 'New Chat',
    preview: 'Start a new procurement query.',
    message_count: 0,
    updated_at: new Date().toISOString(),
    isDraft: true,
  };
}

function draftStorageKey(email) {
  return `${DRAFT_CHATS_KEY}:${email || 'guest'}`;
}

function readDraftChats(email) {
  return readJson(draftStorageKey(email), []);
}

function writeDraftChats(email, drafts) {
  if (!email) return;
  window.localStorage.setItem(draftStorageKey(email), JSON.stringify(drafts));
}

function mergeDraftChats(...draftLists) {
  const merged = new Map();
  draftLists.flat().forEach((chat) => {
    if (!chat?.chat_id) return;
    merged.set(chat.chat_id, chat);
  });
  return Array.from(merged.values()).sort(
    (a, b) => new Date(b.updated_at || 0).getTime() - new Date(a.updated_at || 0).getTime()
  );
}

function mapMessages(messages = []) {
  return messages.map((msg, i) => ({
    ...msg,
    id: `${msg.timestamp || 'local'}-${msg.role}-${i}`,
  }));
}

export default function App() {
  const [session, setSession] = useState(() => readJson(SESSION_KEY, null));
  const [view, setView] = useState('chat');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [feedbackByMessage, setFeedbackByMessage] = useState({});
  const { theme, setTheme } = useTheme();
  const { mode: seasonalMode, setMode: setSeasonalMode, activeFestival } = useSeasonal();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => readJson(SIDEBAR_COLLAPSED_KEY, false));
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [chatTitle, setChatTitle] = useState('New Chat');
  const [messages, setMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [chatError, setChatError] = useState('');
  const [authError, setAuthError] = useState('');
  const canAccessAdmin = Boolean(session?.is_admin);

  // Persist session
  useEffect(() => {
    if (session) window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    else window.localStorage.removeItem(SESSION_KEY);
  }, [session]);

  useEffect(() => { window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, JSON.stringify(sidebarCollapsed)); }, [sidebarCollapsed]);

  // Save draft chats
  useEffect(() => {
    if (!session?.email) return;
    const drafts = chats.filter((c) => c.isDraft);
    writeDraftChats(session.email, drafts);
  }, [chats, session?.email]);

  // Sync auth status
  useEffect(() => {
    if (!session?.email) return;
    let cancelled = false;
    async function sync() {
      try {
        const status = await api.getAuthStatus(session.email);
        if (cancelled) return;
        setSession((s) => s ? {
          ...s,
          displayName: status.display_name || '',
          username: status.username || '',
          avatarBase64: status.avatar_base64 || '',
          totpEnabled: status.totp_enabled,
          is_admin: status.is_admin,
        } : s);
        setAuthError('');
      } catch (err) {
        if (cancelled) return;
        if (err.message === 'User not found.') handleLogout();
        else setAuthError(err.message);
      }
    }
    sync();
    return () => { cancelled = true; };
  }, [session?.email]);

  // Load chats on login
  useEffect(() => {
    if (!session?.email) {
      setChats([]); setActiveChatId(null); setMessages([]); setChatTitle('New Chat');
      return;
    }
    refreshChats();
  }, [session?.email]);

  // Load messages when active chat changes
  useEffect(() => {
    if (!session?.email || !activeChatId) return;
    const activeChat = chats.find((c) => c.chat_id === activeChatId);
    if (activeChat?.isDraft) {
      setMessages([]); setChatTitle(activeChat.title); setChatError('');
      return;
    }
    let cancelled = false;
    async function load() {
      setChatLoading(true);
      try {
        const data = await api.getChat(activeChatId, session.email);
        if (cancelled) return;
        setMessages(mapMessages(data.messages));
        setChatTitle(data.title || 'New Chat');
        setChatError('');
      } catch (err) {
        if (!cancelled) setChatError(err.message);
      } finally {
        if (!cancelled) setChatLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [activeChatId, session?.email, chats]);

  async function refreshChats(preferredChatId = null) {
    if (!session?.email) return;
    try {
      const data = await api.listChats(session.email);
      const serverChats = data.chats || [];
      const storedDrafts = readDraftChats(session.email);
      const inMemoryDrafts = chats.filter((c) => c.isDraft);
      const draftChats = mergeDraftChats(storedDrafts, inMemoryDrafts).filter(
        (d) => !serverChats.some((c) => c.chat_id === d.chat_id)
      );
      const merged = [...draftChats, ...serverChats];
      setChats(merged);
      const next = preferredChatId
        || (activeChatId && merged.some((c) => c.chat_id === activeChatId) ? activeChatId : null)
        || merged[0]?.chat_id || null;
      if (next) setActiveChatId(next);
      else {
        const draft = createDraftChat();
        setChats([draft]); setActiveChatId(draft.chat_id); setMessages([]); setChatTitle(draft.title);
      }
    } catch (err) { setChatError(err.message); }
  }

  function handleAuthenticated(s) {
    setSession(s); setView('chat'); setChats([]); setMessages([]);
    setChatTitle('New Chat'); setActiveChatId(null); setChatError(''); setAuthError(''); setSettingsOpen(false); setProfileOpen(false);
  }

  function handleLogout() {
    setSession(null); setView('chat'); setChats([]); setMessages([]);
    setChatTitle('New Chat'); setActiveChatId(null); setChatError(''); setAuthError(''); setSettingsOpen(false); setProfileOpen(false);
  }

  function handleNewChat() {
    const draft = createDraftChat();
    setChats((cur) => mergeDraftChats([draft], cur));
    setActiveChatId(draft.chat_id); setMessages([]); setChatTitle(draft.title);
    setView('chat'); setChatError('');
  }

  async function handleSendMessage(text) {
    if (!session?.email || !text.trim() || sending) return;
    const chatId = activeChatId || createDraftChat().chat_id;
    const optimistic = [...messages, {
      id: `local-user-${Date.now()}`, role: 'user', content: text.trim(), timestamp: new Date().toISOString(),
    }];
    if (!activeChatId) setActiveChatId(chatId);
    setMessages(optimistic);
    setChatTitle((t) => t === 'New Chat' ? text.trim().slice(0, 60) : t);
    setSending(true); setChatError('');
    try {
      const data = await api.sendMessage(chatId, { user: session.email, message: text.trim() });
      setMessages(mapMessages(data.messages));
      setChatTitle(data.chat?.title || text.trim().slice(0, 60));
      setChats((cur) => {
        const summary = data.chat || {
          chat_id: chatId, title: text.trim().slice(0, 60), preview: text.trim().slice(0, 120),
          message_count: data.messages.length, updated_at: new Date().toISOString(),
        };
        return [summary, ...cur.filter((c) => c.chat_id !== chatId)];
      });
      setActiveChatId(chatId);
      await refreshChats(chatId);
    } catch (err) { setChatError(err.message); }
    finally { setSending(false); }
  }

  async function handleRegenerateResponse() {
    if (!session?.email || !activeChatId || sending) return;
    setSending(true);
    setChatError('');
    try {
      const data = await api.regenerateResponse(activeChatId, session.email);
      setMessages(mapMessages(data.messages));
      setChatTitle(data.chat?.title || chatTitle);
      setChats((cur) => {
        if (!data.chat) return cur;
        return [data.chat, ...cur.filter((item) => item.chat_id !== data.chat.chat_id)];
      });
      await refreshChats(activeChatId);
    } catch (err) {
      setChatError(err.message);
    } finally {
      setSending(false);
    }
  }

  async function handleFeedback(messageId, type) {
    if (!session?.email || !activeChatId || !messageId) return;
    try {
      await api.sendFeedback({
        user: session.email,
        chatId: activeChatId,
        messageId,
        type,
      });
      setFeedbackByMessage((current) => ({
        ...current,
        [`${activeChatId}:${messageId}`]: type,
      }));
    } catch (err) {
      setChatError(err.message);
    }
  }

  async function handleExportChat() {
    if (!session?.email || !activeChatId || exporting) return;
    setExporting(true);
    setChatError('');
    try {
      const file = await api.exportChatPdf(activeChatId, session.email);
      const url = window.URL.createObjectURL(file);
      const link = document.createElement('a');
      const fallbackTitle = (chatTitle || 'procurebuddy-chat').replace(/[^a-zA-Z0-9_ -]/g, '').trim() || 'procurebuddy-chat';
      link.href = url;
      link.download = `${fallbackTitle.replace(/\s+/g, '-').toLowerCase()}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setChatError(err.message);
    } finally {
      setExporting(false);
    }
  }

  if (!session) return <LoginPage onAuthenticated={handleAuthenticated} />;

  return (
    <Layout
      chatTitle={chatTitle}
      userEmail={session.email}
      userDisplayName={session.displayName}
      username={session.username}
      avatarBase64={session.avatarBase64}
      activeView={view}
      setActiveView={setView}
      chats={chats}
      selectedChatId={activeChatId}
      sidebarCollapsed={sidebarCollapsed}
      onToggleCollapsed={() => setSidebarCollapsed((c) => !c)}
      canAccessAdmin={canAccessAdmin}
      onSelectChat={(id) => { setActiveChatId(id); setView('chat'); }}
      onNewChat={handleNewChat}
      onOpenSettings={() => {
        setView('chat');
        setProfileOpen(false);
        setSettingsOpen(true);
      }}
      onOpenProfile={() => {
        setView('chat');
        setSettingsOpen(false);
        setProfileOpen(true);
      }}
      onLogout={handleLogout}
    >
      {view === 'chat' && (
        <ChatView
          title={chatTitle}
          messages={messages}
          loading={chatLoading}
          sending={sending}
          exporting={exporting}
          error={chatError || authError}
          onSend={handleSendMessage}
          onNewChat={handleNewChat}
          onRegenerate={handleRegenerateResponse}
          onFeedback={handleFeedback}
          onExport={handleExportChat}
          feedbackByMessage={feedbackByMessage}
          activeChatId={activeChatId}
        />
      )}
      {settingsOpen && (
        <SettingsModal onClose={() => setSettingsOpen(false)}>
          <SettingsView
            theme={theme}
            setTheme={setTheme}
            seasonalMode={seasonalMode}
            setSeasonalMode={setSeasonalMode}
            activeFestival={activeFestival}
            session={session}
            onSessionUpdate={(changes) => setSession((s) => ({ ...s, ...changes }))}
          />
        </SettingsModal>
      )}
      {profileOpen && (
        <ProfileModal
          session={session}
          onClose={() => setProfileOpen(false)}
          onSaved={(profile) => {
            setSession((current) => current ? { ...current, ...profile } : current);
            setProfileOpen(false);
          }}
        />
      )}
    </Layout>
  );
}
