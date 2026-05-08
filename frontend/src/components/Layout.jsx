import { useEffect, useState } from 'react';
import Sidebar from '../Sidebar';
import Topbar from './Topbar';
import { cn } from './ui';
import { useSeasonal } from '../context/SeasonalContext';

export default function Layout({
  children,
  chatTitle,
  userEmail,
  userDisplayName,
  username,
  avatarBase64,
  activeView,
  setActiveView,
  chats,
  selectedChatId,
  sidebarCollapsed,
  onToggleCollapsed,
  canAccessAdmin,
  onSelectChat,
  onNewChat,
  onOpenProfile,
  onOpenSettings,
  onLogout,
}) {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [copyrightYear, setCopyrightYear] = useState(() => new Date().getFullYear());
  const [dismissedAnnouncementId, setDismissedAnnouncementId] = useState('');
  const { announcement } = useSeasonal();

  useEffect(() => {
    setMobileSidebarOpen(false);
  }, [activeView, selectedChatId]);

  useEffect(() => {
    const now = new Date();
    const nextYear = new Date(now.getFullYear() + 1, 0, 1, 0, 0, 5);
    const timeout = window.setTimeout(() => {
      setCopyrightYear(new Date().getFullYear());
    }, nextYear.getTime() - now.getTime());

    return () => window.clearTimeout(timeout);
  }, [copyrightYear]);

  useEffect(() => {
    setDismissedAnnouncementId('');
  }, [userEmail, announcement?.id]);

  const visibleAnnouncement = announcement && announcement.id !== dismissedAnnouncementId ? announcement : null;

  return (
    <div className="relative h-screen overflow-hidden">
      <div className="relative z-10 flex h-full min-h-0">
        <Sidebar
          activeView={activeView}
          setActiveView={setActiveView}
          chats={chats}
          selectedChatId={selectedChatId}
          collapsed={sidebarCollapsed}
          mobileOpen={mobileSidebarOpen}
          onCloseMobile={() => setMobileSidebarOpen(false)}
          onToggleCollapsed={onToggleCollapsed}
          canAccessAdmin={canAccessAdmin}
          onSelectChat={onSelectChat}
          onNewChat={onNewChat}
          onOpenProfile={onOpenProfile}
          onOpenSettings={onOpenSettings}
          onLogout={onLogout}
          userEmail={userEmail}
          userDisplayName={userDisplayName}
          username={username}
          avatarBase64={avatarBase64}
        />

        <div className="relative flex min-w-0 flex-1 flex-col min-h-0 overflow-hidden">
          <Topbar
            activeView={activeView}
            chatTitle={chatTitle}
            onOpenSidebar={() => setMobileSidebarOpen(true)}
          />

          {visibleAnnouncement && (
            <div className="px-4 pt-3 sm:px-5">
              <div className="mx-auto w-full max-w-[1248px]">
                <div className="rounded-[24px] border border-white/20 bg-[color:var(--card-bg)] px-4 py-3 shadow-[var(--shadow-panel)] backdrop-blur-[var(--panel-blur)]">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-sm font-semibold text-[color:var(--text-primary)]">{visibleAnnouncement.title}</div>
                      <div className="mt-1 space-y-1">
                        {visibleAnnouncement.messages.map((message) => (
                          <div key={message} className="text-sm leading-6 text-[color:var(--text-secondary)]">
                            {message}
                          </div>
                        ))}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-tertiary)] transition duration-200 hover:border-[color:var(--border-strong)] hover:text-[color:var(--text-primary)]"
                      onClick={() => setDismissedAnnouncementId(visibleAnnouncement.id)}
                      aria-label="Dismiss seasonal announcement"
                    >
                      <CloseIcon />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          <main
            className={cn(
              'flex-1 px-4 pb-3 pt-3 sm:px-5 sm:pb-4',
              activeView === 'chat' ? 'min-h-0 overflow-hidden' : 'overflow-y-auto'
            )}
          >
            <div
              className={cn(
                'mx-auto min-h-full w-full',
                activeView === 'chat' ? 'h-full max-w-[1480px]' : 'max-w-[1180px]'
              )}
            >
              {children}
            </div>
          </main>

          <footer className="border-t border-[color:var(--border-soft)] px-4 py-3 sm:px-5">
            <div className="mx-auto flex w-full max-w-[1248px] flex-col gap-1 text-[11px] leading-5 text-[color:var(--text-tertiary)] sm:flex-row sm:items-center sm:justify-between">
              <div>
                Disclaimer: This is AI-generated guidance. Do not rely on it blindly, and please cross-check important
                procurement or compliance decisions before taking action.
              </div>
              <div className="shrink-0">Copyright (c) {copyrightYear} ProcureBuddy. All rights reserved.</div>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M5 5L15 15M15 5L5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}
