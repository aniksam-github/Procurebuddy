import { useEffect, useState } from 'react';
import LoginPage from './LoginPage';
import Sidebar from './Sidebar';
import FestiveOverlay from './FestiveOverlay';
import { ChatView, SettingsView, AdminView } from './Views';
import { api } from './api';
import { getFestivalContext } from './festivalThemes';
import './dashboard.css';

const SESSION_KEY = 'procurebuddy-session';
const ACCENT_KEY = 'procurebuddy-accent';
const THEME_KEY = 'procurebuddy-theme';
const DRAFT_CHATS_KEY = 'procurebuddy-drafts';
const FESTIVE_MODE_KEY = 'procurebuddy-festive-mode';
const SIDEBAR_COLLAPSED_KEY = 'procurebuddy-sidebar-collapsed';
const DEFAULT_ACCENT = '#b24b7d';
const LEGACY_ACCENT = '#0f766e';
const DEFAULT_THEME = 'system';

function clampChannel(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function hexToRgb(hex) {
  const clean = (hex || '').replace('#', '');
  if (clean.length !== 6) {
    return null;
  }

  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

function rgbToHex({ r, g, b }) {
  return `#${[r, g, b].map((channel) => clampChannel(channel).toString(16).padStart(2, '0')).join('')}`;
}

function mixHex(baseHex, targetHex, amount) {
  const base = hexToRgb(baseHex);
  const target = hexToRgb(targetHex);

  if (!base || !target) {
    return baseHex;
  }

  return rgbToHex({
    r: base.r + (target.r - base.r) * amount,
    g: base.g + (target.g - base.g) * amount,
    b: base.b + (target.b - base.b) * amount,
  });
}

function withAlpha(hex, alpha) {
  const rgb = hexToRgb(hex);
  if (!rgb) {
    return `rgba(178, 75, 125, ${alpha})`;
  }
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

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
  if (!email) {
    return;
  }
  window.localStorage.setItem(draftStorageKey(email), JSON.stringify(drafts));
}

function mergeDraftChats(...draftLists) {
  const merged = new Map();

  draftLists.flat().forEach((chat) => {
    if (!chat?.chat_id) {
      return;
    }
    merged.set(chat.chat_id, chat);
  });

  return Array.from(merged.values()).sort(
    (left, right) => new Date(right.updated_at || 0).getTime() - new Date(left.updated_at || 0).getTime()
  );
}

function mapMessages(messages = []) {
  return messages.map((message, index) => ({
    ...message,
    id: `${message.timestamp || 'local'}-${message.role}-${index}`,
  }));
}

export default function App() {
  const [session, setSession] = useState(() => readJson(SESSION_KEY, null));
  const [view, setView] = useState('chat');
  const [accent, setAccent] = useState(() => {
    const storedAccent = window.localStorage.getItem(ACCENT_KEY);
    if (!storedAccent || storedAccent === LEGACY_ACCENT) {
      return DEFAULT_ACCENT;
    }
    return storedAccent;
  });
  const [theme, setTheme] = useState(() => window.localStorage.getItem(THEME_KEY) || DEFAULT_THEME);
  const [festiveMode, setFestiveMode] = useState(() => window.localStorage.getItem(FESTIVE_MODE_KEY) || 'auto');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => readJson(SIDEBAR_COLLAPSED_KEY, false)
  );
  const [now, setNow] = useState(() => new Date());
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [chatTitle, setChatTitle] = useState('New Chat');
  const [messages, setMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState('');
  const [authError, setAuthError] = useState('');
  const activeFestival = getFestivalContext(now, festiveMode);
  const effectiveAccent = activeFestival?.palette?.accent || accent;
  const canAccessAdmin = Boolean(session?.is_admin);

  useEffect(() => {
    if (session) {
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    } else {
      window.localStorage.removeItem(SESSION_KEY);
    }
  }, [session]);

  useEffect(() => {
    window.localStorage.setItem(FESTIVE_MODE_KEY, festiveMode);
  }, [festiveMode]);

  useEffect(() => {
    window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, JSON.stringify(sidebarCollapsed));
  }, [sidebarCollapsed]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60 * 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!session?.email) {
      return;
    }

    const drafts = chats.filter((chat) => chat.isDraft);
    writeDraftChats(session.email, drafts);
  }, [chats, session?.email]);

  useEffect(() => {
    window.localStorage.setItem(ACCENT_KEY, accent);
    const root = document.documentElement.style;
    const derivedBrandStops = [
      mixHex(effectiveAccent, '#ffffff', 0.18),
      effectiveAccent,
      mixHex(effectiveAccent, '#000000', 0.08),
      mixHex(effectiveAccent, '#ffffff', 0.3),
    ];
    const brandStops = activeFestival?.palette?.brand || derivedBrandStops;

    root.setProperty('--accent', effectiveAccent);
    root.setProperty('--brand-a', brandStops[0]);
    root.setProperty('--brand-b', brandStops[1]);
    root.setProperty('--brand-c', brandStops[2]);
    root.setProperty('--brand-d', brandStops[3]);
    root.setProperty(
      '--brand-gradient',
      `linear-gradient(135deg, ${brandStops[0]}, ${brandStops[1]} 48%, ${brandStops[2]} 78%, ${brandStops[3]})`
    );

    root.setProperty('--accent-light', withAlpha(effectiveAccent, 0.14));
    root.setProperty('--accent-glow', withAlpha(effectiveAccent, 0.26));
    root.setProperty('--scene-a', activeFestival?.palette?.scene?.[0] || withAlpha(brandStops[0], 0.16));
    root.setProperty('--scene-b', activeFestival?.palette?.scene?.[1] || withAlpha(brandStops[2], 0.14));
    root.setProperty('--scene-c', activeFestival?.palette?.scene?.[2] || withAlpha(brandStops[3], 0.12));
  }, [accent, activeFestival, effectiveAccent]);

  useEffect(() => {
    window.localStorage.setItem(THEME_KEY, theme);
    const root = document.documentElement.style;
    const isDark =
      theme === 'dark' ||
      (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

    if (isDark) {
      root.setProperty('--bg', '#2d2025');
      root.setProperty('--surface', 'rgba(71, 52, 60, 0.84)');
      root.setProperty('--card-bg', 'rgba(78, 57, 66, 0.92)');
      root.setProperty('--sidebar-bg', 'rgba(60, 43, 50, 0.9)');
      root.setProperty('--input-bg', 'rgba(88, 65, 74, 0.9)');
      root.setProperty('--border', 'rgba(226, 161, 182, 0.18)');
      root.setProperty('--border-hover', 'rgba(231, 182, 125, 0.3)');
      root.setProperty('--text-primary', '#fff2ec');
      root.setProperty('--text-secondary', '#f2dfd7');
      root.setProperty('--text-muted', '#dfc1b7');
      root.setProperty('--shell-1', '#271c21');
      root.setProperty('--shell-2', '#34262d');
      root.setProperty('--shell-3', '#20171b');
      root.setProperty('--grid-line', 'rgba(255, 232, 223, 0.05)');
      root.setProperty('--chrome-bg', 'rgba(43, 31, 37, 0.76)');
      root.setProperty('--chrome-border', 'rgba(255, 233, 225, 0.1)');
      root.setProperty('--sidebar-edge', 'rgba(255, 233, 225, 0.08)');
      root.setProperty(
        '--stage-bg',
        'linear-gradient(145deg, rgba(65, 47, 55, 0.96), rgba(82, 60, 70, 0.92) 56%, rgba(60, 46, 74, 0.94)), radial-gradient(circle at top right, rgba(215, 138, 87, 0.12), transparent 28%)'
      );
      root.setProperty('--stage-title', '#fff3ec');
      root.setProperty('--stage-copy', '#efd8cf');
      root.setProperty('--orb-a-fill', 'rgba(88, 66, 75, 0.82)');
      root.setProperty(
        '--orb-a-shadow',
        'inset -16px -18px 30px rgba(217, 108, 141, 0.14), 20px 20px 40px rgba(10, 7, 8, 0.22)'
      );
      root.setProperty('--orb-b-fill', 'rgba(76, 58, 66, 0.86)');
      root.setProperty(
        '--orb-b-shadow',
        'inset -14px -14px 26px rgba(143, 175, 112, 0.12), 18px 18px 36px rgba(10, 7, 8, 0.2)'
      );
      root.setProperty('--assistant-bubble-bg', 'rgba(73, 55, 63, 0.92)');
      root.setProperty('--assistant-bubble-border', 'rgba(255, 233, 225, 0.08)');
      root.setProperty('--input-shell-bg', 'rgba(38, 28, 33, 0.7)');
      root.setProperty('--input-wrap-bg', 'rgba(67, 50, 58, 0.92)');
      root.setProperty('--input-wrap-border', 'rgba(255, 233, 225, 0.08)');
      root.setProperty(
        '--empty-shell-bg',
        'radial-gradient(circle at top, rgba(217, 108, 141, 0.12), transparent 34%), linear-gradient(145deg, rgba(67, 50, 58, 0.98), rgba(91, 67, 77, 0.96) 52%, rgba(74, 57, 88, 0.96))'
      );
      root.setProperty('--empty-title', '#fff5ef');
      root.setProperty('--empty-copy', 'rgba(255, 237, 228, 0.84)');
      root.setProperty(
        '--admin-status-bg',
        'linear-gradient(145deg, rgba(68, 50, 58, 0.96), rgba(84, 62, 73, 0.94))'
      );
      root.setProperty('--admin-status-pill-bg', 'rgba(255, 236, 228, 0.08)');
      root.setProperty('--admin-status-pill-border', 'rgba(255, 233, 225, 0.1)');
    } else {
      root.setProperty('--bg', '#f2ebe5');
      root.setProperty('--surface', 'rgba(255, 248, 243, 0.8)');
      root.setProperty('--card-bg', 'rgba(255, 250, 246, 0.9)');
      root.setProperty('--sidebar-bg', 'rgba(244, 234, 227, 0.86)');
      root.setProperty('--input-bg', 'rgba(255, 251, 247, 0.94)');
      root.setProperty('--border', 'rgba(178, 75, 125, 0.16)');
      root.setProperty('--border-hover', 'rgba(198, 125, 92, 0.28)');
      root.setProperty('--text-primary', '#3f2a35');
      root.setProperty('--text-secondary', '#6b5058');
      root.setProperty('--text-muted', '#9c7f7b');
      root.setProperty('--shell-1', '#e9ddd6');
      root.setProperty('--shell-2', '#f1e7e1');
      root.setProperty('--shell-3', '#e6ddd3');
      root.setProperty('--grid-line', 'rgba(255, 245, 238, 0.26)');
      root.setProperty('--chrome-bg', 'rgba(250, 242, 236, 0.66)');
      root.setProperty('--chrome-border', 'rgba(255, 247, 242, 0.78)');
      root.setProperty('--sidebar-edge', 'rgba(255, 247, 242, 0.72)');
      root.setProperty(
        '--stage-bg',
        'linear-gradient(140deg, rgba(255, 249, 245, 0.9), rgba(244, 228, 221, 0.84) 54%, rgba(241, 232, 243, 0.84)), radial-gradient(circle at top right, rgba(157, 118, 200, 0.1), transparent 26%)'
      );
      root.setProperty('--stage-title', '#553744');
      root.setProperty('--stage-copy', '#765b64');
      root.setProperty('--orb-a-fill', 'rgba(255, 250, 246, 0.92)');
      root.setProperty(
        '--orb-a-shadow',
        'inset -16px -18px 30px rgba(217, 108, 141, 0.12), 20px 20px 40px rgba(92, 58, 65, 0.14)'
      );
      root.setProperty('--orb-b-fill', 'rgba(255, 251, 247, 0.95)');
      root.setProperty(
        '--orb-b-shadow',
        'inset -14px -14px 26px rgba(143, 175, 112, 0.14), 18px 18px 36px rgba(92, 58, 65, 0.12)'
      );
      root.setProperty('--assistant-bubble-bg', 'rgba(255, 249, 245, 0.9)');
      root.setProperty('--assistant-bubble-border', 'rgba(255, 244, 239, 0.92)');
      root.setProperty('--input-shell-bg', 'rgba(247, 238, 232, 0.56)');
      root.setProperty('--input-wrap-bg', 'rgba(255, 250, 246, 0.92)');
      root.setProperty('--input-wrap-border', 'rgba(255, 246, 241, 0.94)');
      root.setProperty(
        '--empty-shell-bg',
        'radial-gradient(circle at top, rgba(255, 220, 206, 0.2), transparent 36%), linear-gradient(145deg, rgba(125, 89, 101, 0.94), rgba(176, 116, 100, 0.9) 52%, rgba(134, 106, 151, 0.92))'
      );
      root.setProperty('--empty-title', '#fff5ef');
      root.setProperty('--empty-copy', 'rgba(255, 244, 236, 0.82)');
      root.setProperty(
        '--admin-status-bg',
        'linear-gradient(145deg, rgba(255, 248, 243, 0.92), rgba(247, 236, 228, 0.86))'
      );
      root.setProperty('--admin-status-pill-bg', 'rgba(255, 251, 247, 0.82)');
      root.setProperty('--admin-status-pill-border', 'rgba(255, 245, 240, 0.9)');
    }

    document.documentElement.style.colorScheme = isDark ? 'dark' : 'light';
    document.documentElement.style.background = isDark ? '#2d2025' : '#f2ebe5';
    document.body.style.background = isDark ? '#2d2025' : '#f2ebe5';
  }, [theme]);

  useEffect(() => {
    if (!session?.email) {
      return;
    }

    let cancelled = false;

    async function syncSession() {
      try {
        const status = await api.getAuthStatus(session.email);
        if (cancelled) {
          return;
        }
        setSession((current) =>
          current
            ? { ...current, totpEnabled: status.totp_enabled, is_admin: status.is_admin }
            : current
        );
        setAuthError('');
      } catch (error) {
        if (cancelled) {
          return;
        }
        if (error.message === 'User not found.') {
          handleLogout();
        } else {
          setAuthError(error.message);
        }
      }
    }

    syncSession();
    return () => {
      cancelled = true;
    };
  }, [session?.email]);

  useEffect(() => {
    if (view === 'admin' && !canAccessAdmin) {
      setView('chat');
    }
  }, [canAccessAdmin, view]);

  useEffect(() => {
    if (!session?.email) {
      setChats([]);
      setActiveChatId(null);
      setMessages([]);
      setChatTitle('New Chat');
      return;
    }

    refreshChats();
  }, [session?.email]);

  useEffect(() => {
    if (!session?.email || !activeChatId) {
      return;
    }

    const activeChat = chats.find((chat) => chat.chat_id === activeChatId);
    if (activeChat?.isDraft) {
      setMessages([]);
      setChatTitle(activeChat.title);
      setChatError('');
      return;
    }

    let cancelled = false;

    async function loadChat() {
      setChatLoading(true);
      try {
        const data = await api.getChat(activeChatId, session.email);
        if (cancelled) {
          return;
        }
        setMessages(mapMessages(data.messages));
        setChatTitle(data.title || 'New Chat');
        setChatError('');
      } catch (error) {
        if (!cancelled) {
          setChatError(error.message);
        }
      } finally {
        if (!cancelled) {
          setChatLoading(false);
        }
      }
    }

    loadChat();
    return () => {
      cancelled = true;
    };
  }, [activeChatId, session?.email, chats]);

  async function refreshChats(preferredChatId = null) {
    if (!session?.email) {
      return;
    }

    try {
      const data = await api.listChats(session.email);
      const serverChats = data.chats || [];
      const storedDrafts = readDraftChats(session.email);
      const inMemoryDrafts = chats.filter((chat) => chat.isDraft);
      const draftChats = mergeDraftChats(storedDrafts, inMemoryDrafts).filter(
        (draft) => !serverChats.some((chat) => chat.chat_id === draft.chat_id)
      );
      const mergedChats = [...draftChats, ...serverChats];

      setChats(mergedChats);

      const nextChatId =
        preferredChatId ||
        (activeChatId && mergedChats.some((chat) => chat.chat_id === activeChatId) ? activeChatId : null) ||
        mergedChats[0]?.chat_id ||
        null;

      if (nextChatId) {
        setActiveChatId(nextChatId);
      } else {
        const draft = createDraftChat();
        setChats([draft]);
        setActiveChatId(draft.chat_id);
        setMessages([]);
        setChatTitle(draft.title);
      }
    } catch (error) {
      setChatError(error.message);
    }
  }

  function handleAuthenticated(nextSession) {
    setSession(nextSession);
    setView('chat');
    setChats([]);
    setMessages([]);
    setChatTitle('New Chat');
    setActiveChatId(null);
    setChatError('');
    setAuthError('');
  }

  function handleLogout() {
    setSession(null);
    setView('chat');
    setChats([]);
    setMessages([]);
    setChatTitle('New Chat');
    setActiveChatId(null);
    setChatError('');
    setAuthError('');
  }

  function handleNewChat() {
    const draft = createDraftChat();
    setChats((current) => mergeDraftChats([draft], current));
    setActiveChatId(draft.chat_id);
    setMessages([]);
    setChatTitle(draft.title);
    setView('chat');
    setChatError('');
  }

  async function handleSendMessage(text) {
    if (!session?.email || !text.trim() || sending) {
      return;
    }

    const chatId = activeChatId || createDraftChat().chat_id;
    const optimisticMessages = [
      ...messages,
      {
        id: `local-user-${Date.now()}`,
        role: 'user',
        content: text.trim(),
        timestamp: new Date().toISOString(),
      },
    ];

    if (!activeChatId) {
      setActiveChatId(chatId);
    }

    setMessages(optimisticMessages);
    setChatTitle((current) => (current === 'New Chat' ? text.trim().slice(0, 60) : current));
    setSending(true);
    setChatError('');

    try {
      const data = await api.sendMessage(chatId, {
        user: session.email,
        message: text.trim(),
      });

      setMessages(mapMessages(data.messages));
      setChatTitle(data.chat?.title || text.trim().slice(0, 60));
      setChats((current) => {
        const nextSummary = data.chat || {
          chat_id: chatId,
          title: text.trim().slice(0, 60),
          preview: text.trim().slice(0, 120),
          message_count: data.messages.length,
          updated_at: new Date().toISOString(),
        };
        const withoutCurrent = current.filter((chat) => chat.chat_id !== chatId);
        return [nextSummary, ...withoutCurrent];
      });
      setActiveChatId(chatId);
      await refreshChats(chatId);
    } catch (error) {
      setChatError(error.message);
    } finally {
      setSending(false);
    }
  }

  if (!session) {
    return <LoginPage onAuthenticated={handleAuthenticated} />;
  }

  return (
    <div className="dashboard">
      <FestiveOverlay festival={activeFestival} />
      <Sidebar
        activeView={view}
        setActiveView={setView}
        chats={chats}
        selectedChatId={activeChatId}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((current) => !current)}
        canAccessAdmin={canAccessAdmin}
        onSelectChat={(chatId) => {
          setActiveChatId(chatId);
          setView('chat');
        }}
        onNewChat={handleNewChat}
        onLogout={handleLogout}
        userEmail={session.email}
      />
      <div className="main">
        {view === 'chat' && (
          <ChatView
            title={chatTitle}
            messages={messages}
            loading={chatLoading}
            sending={sending}
            error={chatError || authError}
            onSend={handleSendMessage}
            onNewChat={handleNewChat}
          />
        )}
        {view === 'settings' && (
          <SettingsView
            accent={accent}
            setAccent={setAccent}
            theme={theme}
            setTheme={setTheme}
            festiveMode={festiveMode}
            setFestiveMode={setFestiveMode}
            activeFestival={activeFestival}
            session={session}
            onSessionUpdate={(changes) => setSession((current) => ({ ...current, ...changes }))}
          />
        )}
        {view === 'admin' && canAccessAdmin && <AdminView sessionEmail={session.email} />}
      </div>
    </div>
  );
}
