import { useEffect, useState } from 'react';
import Sidebar from '../Sidebar';
import Topbar from './Topbar';
import { cn } from './ui';

export default function Layout({
  children,
  chatTitle,
  userEmail,
  activeView,
  setActiveView,
  chats,
  selectedChatId,
  sidebarCollapsed,
  onToggleCollapsed,
  canAccessAdmin,
  onSelectChat,
  onNewChat,
  onOpenSettings,
  onLogout,
}) {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  useEffect(() => {
    setMobileSidebarOpen(false);
  }, [activeView, selectedChatId]);

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
          onOpenSettings={onOpenSettings}
          onLogout={onLogout}
          userEmail={userEmail}
        />

        <div className="relative flex min-w-0 flex-1 flex-col min-h-0 overflow-hidden">
          <Topbar
            activeView={activeView}
            chatTitle={chatTitle}
            onOpenSidebar={() => setMobileSidebarOpen(true)}
          />

          <main
            className={cn(
              'flex-1 px-4 pb-4 pt-3 sm:px-5 sm:pb-5',
              activeView === 'chat' ? 'min-h-0 overflow-hidden' : 'overflow-y-auto'
            )}
          >
            <div
              className={cn(
                'mx-auto min-h-full w-full',
                activeView === 'chat' ? 'h-full max-w-[1500px]' : 'max-w-[1180px]'
              )}
            >
              {children}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
