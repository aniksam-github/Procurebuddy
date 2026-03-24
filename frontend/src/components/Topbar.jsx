import { useMemo } from 'react';
import { IconButton, Panel } from './ui';

const VIEW_META = {
  chat: {
    title: 'Chat workspace',
    description: 'Clear, grounded, and optimized for long answers.',
  },
  settings: {
    title: 'Settings',
    description: 'Theme, UI mode, font, and account preferences.',
  },
  admin: {
    title: 'Knowledge base',
    description: 'Manage source documents and refresh indexing.',
  },
};

export default function Topbar({ activeView, chatTitle, onOpenSidebar }) {
  const meta = useMemo(() => VIEW_META[activeView] || VIEW_META.chat, [activeView]);
  const title = activeView === 'chat' ? chatTitle || 'New chat' : meta.title;
  const subtitle = meta.description;

  return (
    <div className="px-4 pt-4 sm:px-5 lg:hidden">
      <Panel className="px-4 py-3 sm:px-5">
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <IconButton className="lg:hidden" onClick={onOpenSidebar} aria-label="Open sidebar">
              <MenuIcon />
            </IconButton>

            <div className="min-w-0">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
                {meta.title}
              </div>
              <div className="mt-1 truncate text-lg font-semibold tracking-[-0.03em] text-[color:var(--text-primary)]">
                {title}
              </div>
              <div className="mt-1 hidden text-sm text-[color:var(--text-secondary)] sm:block">{subtitle}</div>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
}

function MenuIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M3.333 5.833H16.667M3.333 10H16.667M3.333 14.167H16.667"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}
