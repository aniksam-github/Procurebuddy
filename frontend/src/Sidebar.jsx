import { useMemo, useState } from 'react';

function formatWhen(value) {
  if (!value) {
    return '';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  const diffMs = Date.now() - date.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffHours / 24);

  if (diffHours < 1) {
    return 'Now';
  }
  if (diffHours < 24) {
    return `${diffHours}h`;
  }
  if (diffDays === 1) {
    return '1d';
  }
  if (diffDays < 7) {
    return `${diffDays}d`;
  }

  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function initials(email) {
  return email
    .split('@')[0]
    .split(/[._-]/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() || '')
    .join('') || 'PB';
}

export default function Sidebar({
  activeView,
  setActiveView,
  chats,
  selectedChatId,
  collapsed,
  onToggleCollapsed,
  canAccessAdmin,
  onSelectChat,
  onNewChat,
  onLogout,
  userEmail,
}) {
  const [search, setSearch] = useState('');
  const [showMenu, setShowMenu] = useState(false);

  const filteredChats = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) {
      return chats;
    }

    return chats.filter((chat) =>
      `${chat.title} ${chat.preview || ''}`.toLowerCase().includes(query)
    );
  }, [chats, search]);

  return (
    <div className={`sidebar${collapsed ? ' collapsed' : ''}`}>
      <button
        className="sidebar-collapse-btn"
        type="button"
        onClick={onToggleCollapsed}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? '>>' : '<<'}
      </button>
      <div className="sb-logo">
        <div className="logo-mark">PB</div>
        {!collapsed && (
          <div>
          <div className="sb-logo-name">ProcureBuddy</div>
          <div className="profile-role">CBRI procurement copilot</div>
          </div>
        )}
      </div>

      <button className="sb-btn new-chat" onClick={onNewChat} title="New Chat">
        <span style={{ fontSize: 15, flexShrink: 0 }}>+</span>
        {!collapsed && 'New Chat'}
      </button>

      {!collapsed && (
        <div className="search-box">
          <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Search</span>
          <input
            placeholder="Search chat history"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
      )}

      {!collapsed && <div className="sec-label">Conversations</div>}
      {filteredChats.map((chat) => (
        <button
          key={chat.chat_id}
          className={`sb-btn${activeView === 'chat' && chat.chat_id === selectedChatId ? ' active' : ''}`}
          onClick={() => onSelectChat(chat.chat_id)}
          title={chat.title}
        >
          <span style={{ fontSize: 13, flexShrink: 0 }}>Q</span>
          {!collapsed && (
            <>
              <span style={{ flex: 1, minWidth: 0 }}>
                <span className="chat-row-title">{chat.title}</span>
                <span className="chat-row-preview">{chat.preview || 'No messages yet.'}</span>
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>
                {formatWhen(chat.updated_at)}
              </span>
            </>
          )}
        </button>
      ))}
      {!collapsed && filteredChats.length === 0 && (
        <div className="empty-side-note">No chats match your search yet.</div>
      )}

      <div className="sb-spacer" />
      {canAccessAdmin && (
        <button
          className={`sb-btn${activeView === 'admin' ? ' active' : ''}`}
          onClick={() => setActiveView('admin')}
          title="Knowledge Base"
        >
          <span style={{ fontSize: 13, flexShrink: 0 }}>KB</span>
          {!collapsed && 'Knowledge Base'}
        </button>
      )}
      <button
        className={`sb-btn${activeView === 'settings' ? ' active' : ''}`}
        onClick={() => setActiveView('settings')}
        title="Settings"
      >
        <span style={{ fontSize: 13, flexShrink: 0 }}>S</span>
        {!collapsed && 'Settings'}
      </button>

      <div style={{ position: 'relative' }}>
        {showMenu && (
          <div className="dropdown">
            <div className="dd-item" onClick={() => { setActiveView('settings'); setShowMenu(false); }}>
              Account settings
            </div>
            <div className="dd-item danger" style={{ color: 'var(--danger)' }} onClick={onLogout}>
              Log out
            </div>
          </div>
        )}
        <div className="profile-section" onClick={() => setShowMenu((current) => !current)}>
          <div className="avatar">{initials(userEmail)}</div>
          {!collapsed && (
            <>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="profile-name">{userEmail}</div>
                <div className="profile-role">{chats.length} stored chats</div>
              </div>
              <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>...</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
