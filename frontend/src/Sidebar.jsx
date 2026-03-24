import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Button, Panel, cn } from './components/ui';

function formatWhen(value) {
  if (!value) return 'Now';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Now';
  const diffMs = Date.now() - date.getTime();
  const diffHours = Math.floor(diffMs / 3_600_000);
  const diffDays = Math.floor(diffHours / 24);

  if (diffHours < 1) return 'Now';
  if (diffHours < 24) return `${diffHours}h`;
  if (diffDays < 7) return `${diffDays}d`;

  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function getInitials(email = '') {
  return (
    email
      .split('@')[0]
      .split(/[._-]/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() || '')
      .join('') || 'PB'
  );
}

export default function Sidebar({
  activeView,
  setActiveView,
  chats,
  selectedChatId,
  collapsed,
  mobileOpen,
  onCloseMobile,
  onToggleCollapsed,
  canAccessAdmin,
  onSelectChat,
  onNewChat,
  onOpenSettings,
  onLogout,
  userEmail,
}) {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [showMenu, setShowMenu] = useState(false);

  const filteredChats = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) {
      return chats;
    }

    return chats.filter((chat) =>
      `${chat.title || ''} ${chat.preview || ''}`.toLowerCase().includes(query)
    );
  }, [chats, search]);

  return (
    <>
      <AnimatePresence>
        {mobileOpen && (
          <motion.button
            type="button"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onCloseMobile}
            className="fixed inset-0 z-30 bg-slate-950/35 backdrop-blur-sm lg:hidden"
            aria-label="Close sidebar"
          />
        )}
      </AnimatePresence>

        <motion.aside
          initial={false}
          animate={{ width: collapsed ? 108 : 320 }}
          transition={{ duration: 0.24, ease: [0.4, 0, 0.2, 1] }}
          className={cn(
          'fixed inset-y-0 left-0 z-40 flex h-screen w-[320px] shrink-0 flex-col overflow-hidden p-4 transition duration-200 lg:static lg:h-full lg:min-h-0 lg:z-10',
          mobileOpen
            ? 'translate-x-0 opacity-100 pointer-events-auto'
            : '-translate-x-full opacity-0 pointer-events-none lg:translate-x-0 lg:opacity-100 lg:pointer-events-auto'
          )}
        >
        <Panel className={cn('flex h-full flex-col overflow-hidden py-3', collapsed ? 'px-2' : 'px-3')}>
          <div className={cn('px-2 pb-3', collapsed ? 'flex flex-col items-center gap-2' : 'flex items-center gap-3')}>
            <button
              type="button"
              onClick={() => navigate('/')}
              className="brand-gradient-bg flex h-12 w-12 shrink-0 items-center justify-center rounded-[20px] text-sm font-extrabold text-white shadow-glow transition duration-200 hover:scale-[1.02]"
              title="Go to homepage"
              aria-label="Go to homepage"
            >
              PB
            </button>

            {!collapsed && (
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-semibold text-[color:var(--text-primary)]">
                  ProcureBuddy AI
                </div>
                <div className="text-xs text-[color:var(--text-tertiary)]">Chat-first procurement workspace</div>
              </div>
            )}

            <button
              type="button"
              onClick={onToggleCollapsed}
              className="hidden h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-secondary)] transition duration-200 hover:border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)] hover:text-[color:var(--text-primary)] lg:inline-flex"
              title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              <ChevronIcon collapsed={collapsed} />
            </button>

            <button
              type="button"
              onClick={onCloseMobile}
              className="inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-secondary)] lg:hidden"
              aria-label="Close sidebar"
            >
              <CloseIcon />
            </button>
          </div>

          <div className={cn('px-2 pb-3', collapsed && 'flex justify-center')}>
            <Button
              className={cn('justify-center', collapsed ? 'h-12 w-12 rounded-[20px] px-0' : 'w-full')}
              onClick={onNewChat}
              title="New chat"
              aria-label="New chat"
            >
              <PlusIcon />
              {!collapsed && 'New chat'}
            </Button>
          </div>

          {!collapsed && (
            <div className="px-2 pb-3">
              <input
                className="app-input py-2.5 text-sm"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search conversations"
              />
            </div>
          )}

          {!collapsed && (
            <div className="px-2 pb-2">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
                Conversations
              </div>
            </div>
          )}

          <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-3">
            {filteredChats.map((chat) => {
              const active = activeView === 'chat' && chat.chat_id === selectedChatId;

              return (
                <button
                  key={chat.chat_id}
                  type="button"
                  onClick={() => {
                    onSelectChat(chat.chat_id);
                    onCloseMobile();
                  }}
                  className={cn(
                    'flex w-full items-center gap-3 rounded-[22px] border px-3 py-3 text-left transition duration-200',
                    active
                      ? 'border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]'
                      : 'border-transparent bg-transparent text-[color:var(--text-secondary)] hover:border-[color:var(--border-soft)] hover:bg-[color:var(--card-subtle)] hover:text-[color:var(--text-primary)]',
                    collapsed && 'justify-center px-0'
                  )}
                  title={chat.title}
                >
                  <div
                    className={cn(
                      'h-2.5 w-2.5 shrink-0 rounded-full',
                      active ? 'bg-[color:var(--accent)]' : 'bg-[color:var(--border-strong)]'
                    )}
                  />

                  {!collapsed && (
                    <>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-semibold">{chat.title}</div>
                        <div className="truncate text-xs text-[color:var(--text-tertiary)]">
                          {chat.preview || 'No messages yet'}
                        </div>
                      </div>
                      <div className="text-[11px] font-medium text-[color:var(--text-tertiary)]">
                        {formatWhen(chat.updated_at)}
                      </div>
                    </>
                  )}
                </button>
              );
            })}

            {!collapsed && filteredChats.length === 0 && (
              <div className="rounded-[22px] border border-dashed border-[color:var(--border-soft)] px-4 py-8 text-center text-sm text-[color:var(--text-tertiary)]">
                No conversations match your search.
              </div>
            )}
          </div>

          <div className="border-t border-[color:var(--border-soft)] px-2 pt-3">
            <div className="relative mt-2">
              <button
                type="button"
                onClick={() => setShowMenu((value) => !value)}
                className={cn(
                  'flex w-full items-center gap-3 rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] px-3 py-3 text-left transition duration-200 hover:bg-[color:var(--card-hover)]',
                  collapsed && 'justify-center px-0'
                )} 
              >
                <div className="brand-gradient-bg flex h-11 w-11 shrink-0 items-center justify-center rounded-[18px] text-xs font-bold text-white">
                  {getInitials(userEmail)}
                </div>

                {!collapsed && (
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold text-[color:var(--text-primary)]">
                      {userEmail}
                    </div>
                    <div className="text-xs text-[color:var(--text-tertiary)]">Account menu</div>
                  </div>
                )}
              </button>

              <AnimatePresence>
                {showMenu && !collapsed && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 6 }}
                    transition={{ duration: 0.18 }}
                    className="absolute bottom-full left-0 right-0 mb-2 rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-bg)] p-2 shadow-[var(--shadow-panel)]"
                  >
                    <button
                      type="button"
                      onClick={() => {
                        onOpenSettings();
                        setShowMenu(false);
                        onCloseMobile();
                      }}
                      className="flex w-full items-center rounded-2xl px-3 py-2.5 text-sm text-[color:var(--text-secondary)] transition duration-200 hover:bg-[color:var(--card-subtle)] hover:text-[color:var(--text-primary)]"
                    >
                      Settings
                    </button>
                    <button
                      type="button"
                      onClick={onLogout}
                      className="mt-1 flex w-full items-center rounded-2xl px-3 py-2.5 text-sm text-rose-500 transition duration-200 hover:bg-rose-500/10"
                    >
                      Log out
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </Panel>
      </motion.aside>
    </>
  );
}

function NavButton({ collapsed, active, onClick, icon, label }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={collapsed ? label : undefined}
      aria-label={label}
      className={cn(
        'mt-1 flex w-full items-center gap-3 rounded-[22px] px-3 py-3 text-left text-sm font-medium transition duration-200',
        active
          ? 'bg-[color:var(--accent-soft)] text-[color:var(--accent)]'
          : 'text-[color:var(--text-secondary)] hover:bg-[color:var(--card-subtle)] hover:text-[color:var(--text-primary)]',
        collapsed && 'justify-center px-0'
      )}
    >
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[18px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)]">
        {icon}
      </span>
      {!collapsed && (
        <span className="min-w-0">
          <span className="block truncate">{label}</span>
        </span>
      )}
    </button>
  );
}

function PlusIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M10 4.167V15.833M4.167 10H15.833" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M5 5L15 15M15 5L5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg className="h-4 w-4 text-current" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 6.667A3.333 3.333 0 1 0 10 13.333A3.333 3.333 0 0 0 10 6.667ZM15.833 10C15.833 9.539 15.783 9.09 15.687 8.658L17.292 7.425L15.625 4.542L13.675 5.292C12.992 4.767 12.207 4.372 11.35 4.142L11.042 2.083H7.708L7.4 4.142C6.543 4.372 5.758 4.767 5.075 5.292L3.125 4.542L1.458 7.425L3.063 8.658C2.967 9.09 2.917 9.539 2.917 10C2.917 10.461 2.967 10.91 3.063 11.342L1.458 12.575L3.125 15.458L5.075 14.708C5.758 15.233 6.543 15.628 7.4 15.858L7.708 17.917H11.042L11.35 15.858C12.207 15.628 12.992 15.233 13.675 14.708L15.625 15.458L17.292 12.575L15.687 11.342C15.783 10.91 15.833 10.461 15.833 10Z"
        stroke="currentColor"
        strokeWidth="1.35"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChevronIcon({ collapsed }) {
  return (
    <svg
      className={cn('h-4 w-4 transition-transform duration-200', collapsed && 'rotate-180')}
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M12.5 4.167L6.667 10L12.5 15.833"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
