import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { AnimatePresence, motion } from 'framer-motion';
import { api } from './api';
import { useSeasonal } from './context/SeasonalContext';
import { useTheme } from './context/ThemeContext';
import { Button, Eyebrow, Panel, Pill, cn } from './components/ui';

const PASSWORD_REQUIREMENTS = [
  { key: 'length', label: 'At least 8 characters' },
  { key: 'uppercase', label: 'One uppercase letter' },
  { key: 'lowercase', label: 'One lowercase letter' },
  { key: 'digit', label: 'One number' },
  { key: 'symbol', label: 'One special symbol' },
];

const SAMPLE_PROMPTS = [
  'Show the process for an 8 lakh purchase.',
  'Compare single tender and limited tender in a table.',
  'Summarize committee approval workflow in simple language.',
];

function getPasswordChecks(password) {
  const value = password || '';
  return {
    length: value.length >= 8,
    uppercase: /[A-Z]/.test(value),
    lowercase: /[a-z]/.test(value),
    digit: /\d/.test(value),
    symbol: /[^A-Za-z0-9]/.test(value),
  };
}

function isStrongPassword(password) {
  return Object.values(getPasswordChecks(password)).every(Boolean);
}

function formatUpdatedAt(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

export function ChatView({ title, messages, loading, sending, error, onSend, onNewChat }) {
  const [input, setInput] = useState('');
  const bottomRef = useRef(null);
  const textAreaRef = useRef(null);
  const { uiMode } = useTheme();
  const { mode: seasonalMode, activeFestival } = useSeasonal();
  const festiveActive = seasonalMode === 'always' || (seasonalMode === 'auto' && activeFestival?.id === 'navratri');
  const ChatShell = festiveActive ? 'div' : Panel;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  function resizeTextArea(nextValue) {
    setInput(nextValue);
    const element = textAreaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`;
  }

  function submit() {
    const text = input.trim();
    if (!text || sending) return;
    setInput('');
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto';
    }
    onSend(text);
  }

  return (
    <div className="h-full min-h-0">
      <ChatShell
        className={cn(
          'flex h-full min-h-0 flex-col overflow-hidden',
          festiveActive
            ? 'festive-chat-layout relative z-10'
            : 'chat-shell-panel relative z-20'
        )}
      >
        {error && (
          <div className="mx-5 mt-4 rounded-2xl border border-rose-200/60 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:border-rose-500/20 dark:text-rose-300 sm:mx-6">
            {error}
          </div>
        )}

        <div
          className={cn(
            'flex-1 min-h-0 overflow-y-auto',
            festiveActive ? 'festive-chat-scroll px-3 py-4 sm:px-5' : 'px-5 py-5 sm:px-6'
          )}
        >
          {loading ? (
            <div className="flex h-full items-center justify-center">
              <div
                className={cn(
                  'rounded-[24px] border px-6 py-5 text-sm',
                  festiveActive
                    ? 'festive-chat-surface text-[color:var(--text-secondary)]'
                    : 'border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-secondary)]'
                )}
              >
                Loading conversation...
              </div>
            </div>
          ) : messages.length === 0 ? (
            <EmptyChatState onPromptSelect={resizeTextArea} festive={festiveActive} />
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.22, delay: Math.min(index * 0.02, 0.16) }}
                >
                  <MessageBubble message={message} futuristic={uiMode === 'futuristic'} festive={festiveActive} />
                </motion.div>
              ))}

              <AnimatePresence>
                {sending && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="flex justify-start"
                  >
                    <div
                      className={cn(
                        'max-w-[82%] rounded-[26px] px-5 py-4',
                        festiveActive
                          ? 'festive-message festive-message-bot'
                          : 'border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] shadow-soft'
                      )}
                    >
                      <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
                        ProcureBuddy
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 animate-pulse rounded-full bg-[color:var(--accent)]" />
                        <span className="h-2 w-2 animate-pulse rounded-full bg-[color:var(--accent)] [animation-delay:120ms]" />
                        <span className="h-2 w-2 animate-pulse rounded-full bg-[color:var(--accent)] [animation-delay:240ms]" />
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div
          className={cn(
            'px-5 py-4 sm:px-6',
            festiveActive ? 'relative z-20 border-t border-white/10 bg-transparent' : 'border-t border-[color:var(--border-soft)]'
          )}
        >
          <div
            className={cn(
              'rounded-[28px] p-3',
              festiveActive
                ? 'festive-chat-composer'
                : 'border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] shadow-soft'
            )}
          >
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
              <textarea
                ref={textAreaRef}
                rows={1}
                value={input}
                className="app-textarea min-h-[56px] flex-1 border-none bg-transparent px-3 py-2.5 text-sm leading-7 shadow-none focus:shadow-none"
                placeholder="Ask about process thresholds, approvals, committees, rules, or comparisons..."
                onChange={(event) => resizeTextArea(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    submit();
                  }
                }}
              />
              <Button className="w-full lg:w-auto" onClick={submit} disabled={sending}>
                Send message
                <ArrowUpIcon />
              </Button>
            </div>
            <div className="mt-3 text-xs leading-6 text-[color:var(--text-tertiary)]">
              Disclaimer: This is AI-generated guidance. Do not rely on it blindly, and please cross-check important
              procurement or compliance decisions before taking action.
            </div>
          </div>
        </div>
      </ChatShell>
    </div>
  );
}

export function SettingsView({
  theme,
  setTheme,
  seasonalMode,
  setSeasonalMode,
  activeFestival,
  session,
  onSessionUpdate,
}) {
  const { uiMode, setUiMode, fontFamily, setFontFamily, resolvedTheme, accentColor, setAccentColor, accentOptions } = useTheme();
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [status, setStatus] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [statusError, setStatusError] = useState('');
  const [totpSecret, setTotpSecret] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [qrBase64, setQrBase64] = useState('');
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [activeSection, setActiveSection] = useState(null);

  const sectionOptions = [
    {
      key: 'appearance',
      title: 'Appearance',
      description: 'Theme, accent color, and typography.',
      icon: <PaletteIcon />,
    },
    {
      key: 'workspace',
      title: 'Workspace',
      description: 'UI mode and seasonal presentation.',
      icon: <SparklesIcon />,
    },
    {
      key: 'security',
      title: 'Security',
      description: 'Password and two-factor authentication.',
      icon: <ShieldIcon />,
    },
    ...(session.is_admin
      ? [
          {
            key: 'knowledge',
            title: 'Knowledge base',
            description: 'Upload files and manage indexing.',
            icon: <DatabaseIcon />,
          },
        ]
      : []),
  ];

  const activeOption = sectionOptions.find((option) => option.key === activeSection) || null;

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      setLoadingStatus(true);
      try {
        const data = await api.getAuthStatus(session.email);
        if (!cancelled) {
          setStatus(data);
          setStatusError('');
        }
      } catch (err) {
        if (!cancelled) {
          setStatusError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingStatus(false);
        }
      }
    }

    loadStatus();
    return () => {
      cancelled = true;
    };
  }, [session.email]);

  useEffect(() => {
    if (activeSection === 'knowledge' && !session.is_admin) {
      setActiveSection(null);
    }
  }, [activeSection, session.is_admin]);

  async function handleChangePassword() {
    if (!password.trim()) return;
    if (!isStrongPassword(password.trim())) {
      setStatusError('Choose a stronger password.');
      return;
    }

    setSubmitting(true);
    try {
      const data = await api.changePassword({ email: session.email, new_password: password.trim() });
      setPassword('');
      setStatusMessage(data.message || 'Password updated.');
      setStatusError('');
    } catch (err) {
      setStatusError(err.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSetupTotp() {
    setSubmitting(true);
    try {
      const data = await api.setupTotp({ email: session.email });
      setTotpSecret(data.secret || '');
      setQrBase64(data.qr_base64 || '');
      setStatusError('');
      setStatusMessage('Scan the QR code and verify the generated code.');
    } catch (err) {
      setStatusError(err.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleEnableTotp() {
    if (!totpCode.trim()) return;
    setSubmitting(true);
    try {
      await api.enableTotp({ email: session.email, code: totpCode.trim() });
      onSessionUpdate({ totpEnabled: true });
      setTotpSecret('');
      setTotpCode('');
      setQrBase64('');
      setStatusMessage('Two-factor authentication enabled.');
      setStatusError('');
    } catch (err) {
      setStatusError(err.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDisableTotp() {
    setSubmitting(true);
    try {
      await api.disableTotp({ email: session.email });
      onSessionUpdate({ totpEnabled: false });
      setTotpSecret('');
      setTotpCode('');
      setQrBase64('');
      setStatusMessage('Two-factor authentication disabled.');
      setStatusError('');
    } catch (err) {
      setStatusError(err.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-4">
      {!activeSection ? (
        <div className="mx-auto max-w-[720px] pr-14">
          <h1 className="text-3xl font-semibold tracking-[-0.04em] text-[color:var(--text-primary)]">
            Open one section at a time
          </h1>
          <p className="mt-2 text-sm leading-7 text-[color:var(--text-secondary)]">
            Choose what you want to adjust. We will open that section only, so the settings screen stays compact.
          </p>

          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            {sectionOptions.map((option) => (
              <SettingsLauncherCard
                key={option.key}
                title={option.title}
                description={option.description}
                icon={option.icon}
                onClick={() => setActiveSection(option.key)}
              />
            ))}
          </div>
        </div>
      ) : (
        <>
          <div className="mx-auto max-w-[920px] pr-14">
            <div className="flex justify-end">
              <Button variant="secondary" size="sm" className="h-9 rounded-xl px-3 text-xs" onClick={() => setActiveSection(null)}>
                <BackIcon />
                All settings
              </Button>
            </div>
            <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-[color:var(--text-primary)]">
              {activeOption?.title}
            </h2>
            <p className="mt-2 max-w-2xl text-sm leading-7 text-[color:var(--text-secondary)]">
              {activeOption?.description}
            </p>
          </div>

          {activeSection === 'appearance' && (
            <div className="grid gap-4 xl:grid-cols-2">
              <div className="space-y-4">
                <Panel className="p-5 sm:p-6">
                  <div className="mb-5 flex items-center justify-between">
                    <div>
                      <div className="text-lg font-semibold text-[color:var(--text-primary)]">Theme</div>
                      <div className="text-sm text-[color:var(--text-secondary)]">Light mode is the primary visual reference.</div>
                    </div>
                    <Pill>{resolvedTheme} active</Pill>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    {[
                      {
                        value: 'light',
                        title: 'Light',
                        description: 'Bright, premium, and optimized for long reading sessions.',
                      },
                      {
                        value: 'dark',
                        title: 'Dark',
                        description: 'Deeper surfaces with preserved contrast and calm accents.',
                      },
                      {
                        value: 'system',
                        title: 'System',
                        description: 'Match the device preference automatically.',
                      },
                    ].map((option) => (
                      <ChoiceCard
                        key={option.value}
                        active={theme === option.value}
                        title={option.title}
                        description={option.description}
                        onClick={() => setTheme(option.value)}
                      />
                    ))}
                  </div>
                </Panel>

                <Panel className="p-5 sm:p-6">
                  <div className="mb-5">
                    <div className="text-lg font-semibold text-[color:var(--text-primary)]">Font family</div>
                    <div className="text-sm text-[color:var(--text-secondary)]">Default to Inter, with optional Satoshi or system font.</div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    {[
                      { value: 'inter', title: 'Inter', description: 'Balanced and highly readable.' },
                      { value: 'satoshi', title: 'Satoshi', description: 'A slightly more editorial tone.' },
                      { value: 'system', title: 'System', description: 'Use the platform font stack.' },
                    ].map((option) => (
                      <ChoiceCard
                        key={option.value}
                        active={fontFamily === option.value}
                        title={option.title}
                        description={option.description}
                        onClick={() => setFontFamily(option.value)}
                      />
                    ))}
                  </div>
                </Panel>
              </div>

              <Panel className="p-5 sm:p-6">
                <div className="mb-5">
                  <div className="text-lg font-semibold text-[color:var(--text-primary)]">Accent color</div>
                  <div className="text-sm text-[color:var(--text-secondary)]">
                    Choose the color used for primary buttons, highlights, selections, and active states.
                  </div>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  {Object.entries(accentOptions).map(([value, option]) => (
                    <AccentCard
                      key={value}
                      active={accentColor === value}
                      title={option.label}
                      onClick={() => setAccentColor(value)}
                      stops={[option.light.brandB, option.light.brandA, option.light.brandC]}
                    />
                  ))}
                </div>
              </Panel>
            </div>
          )}

          {activeSection === 'workspace' && (
            <div className="grid gap-4 xl:grid-cols-2">
              <Panel className="p-5 sm:p-6">
                <div className="mb-5">
                  <div className="text-lg font-semibold text-[color:var(--text-primary)]">UI mode</div>
                  <div className="text-sm text-[color:var(--text-secondary)]">
                    Keep it minimal or add a slightly richer futuristic finish.
                  </div>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <ChoiceCard
                    active={uiMode === 'minimal'}
                    title="Minimal"
                    description="Clean, quiet surfaces with almost no visual excess."
                    onClick={() => setUiMode('minimal')}
                  />
                  <ChoiceCard
                    active={uiMode === 'futuristic'}
                    title="Futuristic"
                    description="Slightly stronger glass, glow, and atmosphere without clutter."
                    onClick={() => setUiMode('futuristic')}
                  />
                </div>
              </Panel>

              <Panel className="p-5 sm:p-6">
                <div className="mb-5">
                  <div className="text-lg font-semibold text-[color:var(--text-primary)]">Seasonal layer</div>
                  <div className="text-sm text-[color:var(--text-secondary)]">
                    Keep festive visuals automatic, always on, or fully disabled.
                  </div>
                </div>
                <div className="grid gap-3">
                  {[
                    {
                      value: 'auto',
                      title: 'Auto',
                      description: activeFestival ? `Now showing ${activeFestival.name}.` : 'Detect seasonal context automatically.',
                    },
                    { value: 'always', title: 'Always on', description: 'Force the festive layer on at full intensity.' },
                    { value: 'off', title: 'Off', description: 'Disable festive visuals for a fully neutral shell.' },
                  ].map((option) => (
                    <ChoiceCard
                      key={option.value}
                      active={seasonalMode === option.value}
                      title={option.title}
                      description={option.description}
                      onClick={() => setSeasonalMode(option.value)}
                    />
                  ))}
                </div>
              </Panel>
            </div>
          )}

          {activeSection === 'security' && (
            <div className="mx-auto max-w-3xl space-y-4">
              {(statusError || statusMessage) && (
                <div
                  className={cn(
                    'rounded-[24px] border px-4 py-3 text-sm',
                    statusError
                      ? 'border-rose-200/60 bg-rose-500/10 text-rose-700 dark:border-rose-500/20 dark:text-rose-300'
                      : 'border-emerald-200/60 bg-emerald-500/10 text-emerald-700 dark:border-emerald-500/20 dark:text-emerald-300'
                  )}
                >
                  {statusError || statusMessage}
                </div>
              )}

              <Panel className="p-5 sm:p-6">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <div className="text-lg font-semibold text-[color:var(--text-primary)]">Security</div>
                    <div className="text-sm text-[color:var(--text-secondary)]">Manage password and two-factor protection.</div>
                  </div>
                  <Pill>{session.totpEnabled ? '2FA enabled' : '2FA disabled'}</Pill>
                </div>

                {loadingStatus ? (
                  <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] px-4 py-5 text-sm text-[color:var(--text-secondary)]">
                    Loading account status...
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-4">
                      <div className="text-sm font-semibold text-[color:var(--text-primary)]">{session.email}</div>
                      <div className="mt-1 text-sm text-[color:var(--text-secondary)]">
                        {status?.is_admin ? 'Admin access enabled.' : 'Standard user access.'}
                      </div>
                    </div>

                    <div>
                      <div className="mb-2 text-sm font-medium text-[color:var(--text-primary)]">Change password</div>
                      <div className="relative">
                        <input
                          className="app-input pr-16"
                          type={showPassword ? 'text' : 'password'}
                          value={password}
                          placeholder="Choose a new password"
                          onChange={(event) => {
                            setPassword(event.target.value);
                            setStatusError('');
                          }}
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword((value) => !value)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-sm font-medium text-[color:var(--text-tertiary)]"
                        >
                          {showPassword ? 'Hide' : 'Show'}
                        </button>
                      </div>
                      {password && <PasswordChecklist password={password} />}
                      <Button className="mt-3 w-full" onClick={handleChangePassword} disabled={submitting}>
                        Save password
                      </Button>
                    </div>

                    <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-4">
                      <div className="text-sm font-semibold text-[color:var(--text-primary)]">Two-factor authentication</div>
                      <p className="mt-2 text-sm leading-7 text-[color:var(--text-secondary)]">
                        Add an authenticator step for stronger account protection.
                      </p>

                      {session.totpEnabled ? (
                        <Button variant="secondary" className="mt-4 w-full" onClick={handleDisableTotp} disabled={submitting}>
                          Disable 2FA
                        </Button>
                      ) : !totpSecret ? (
                        <Button className="mt-4 w-full" onClick={handleSetupTotp} disabled={submitting}>
                          Set up 2FA
                        </Button>
                      ) : (
                        <div className="mt-4 space-y-4">
                          {qrBase64 && (
                            <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] p-4">
                              <img
                                src={`data:image/png;base64,${qrBase64}`}
                                alt="TOTP QR code"
                                className="mx-auto h-44 w-44 rounded-2xl border border-[color:var(--border-soft)] bg-white p-2"
                              />
                              <div className="mt-3 break-all text-xs text-[color:var(--text-tertiary)]">{totpSecret}</div>
                            </div>
                          )}
                          <input
                            className="app-input"
                            type="text"
                            inputMode="numeric"
                            value={totpCode}
                            placeholder="Enter the 6-digit code"
                            onChange={(event) => setTotpCode(event.target.value)}
                          />
                          <Button className="w-full" onClick={handleEnableTotp} disabled={submitting}>
                            Verify and enable
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </Panel>
            </div>
          )}

          {activeSection === 'knowledge' && session.is_admin && (
            <div className="space-y-4">
              <Panel className="p-5 sm:p-6">
                <Eyebrow>Knowledge base</Eyebrow>
                <h2 className="mt-4 text-2xl font-semibold tracking-[-0.04em] text-[color:var(--text-primary)]">
                  Admin document operations
                </h2>
                <p className="mt-2 max-w-3xl text-sm leading-7 text-[color:var(--text-secondary)]">
                  Upload source files, trigger indexing, and monitor processing state from inside Settings.
                </p>
              </Panel>

              <AdminView sessionEmail={session.email} embedded />
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function SettingsModal({ children, onClose }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === 'Escape') onClose();
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      <button
        type="button"
        className="absolute inset-0 bg-slate-950/28 backdrop-blur-sm"
        aria-label="Close settings"
        onClick={onClose}
      />
      <Panel className="relative z-10 max-h-[88vh] w-full max-w-[980px] overflow-y-auto p-5 sm:p-6">
        <button
          type="button"
          onClick={onClose}
          className="absolute right-5 top-5 inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-secondary)] transition duration-200 hover:border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)] hover:text-[color:var(--text-primary)]"
          aria-label="Close settings"
        >
          <CloseIcon />
        </button>
        {children}
      </Panel>
    </div>
  );
}

export function AdminView({ sessionEmail, embedded = false }) {
  const [documents, setDocuments] = useState([]);
  const [adminStatus, setAdminStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    loadDocuments();
    loadAdminStatus();
  }, [sessionEmail]);

  async function loadDocuments() {
    setLoading(true);
    try {
      const data = await api.listDocuments(sessionEmail);
      setDocuments(data.documents || []);
      setError('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadAdminStatus() {
    try {
      const data = await api.getAdminStatus(sessionEmail);
      setAdminStatus(data);
    } catch {
      // Keep silent here. The page still works with the document list alone.
    }
  }

  async function handleReindex() {
    setReindexing(true);
    try {
      const data = await api.reindexDocuments(sessionEmail);
      setMessage(data.message || 'Reindex started.');
      setError('');
      await loadAdminStatus();
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setReindexing(false);
    }
  }

  async function handleUpload() {
    if (!selectedFiles.length) return;
    setUploading(true);
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append('files', file));
      const data = await api.uploadDocuments(sessionEmail, formData);
      setMessage(data.message || 'Upload complete.');
      setError('');
      setSelectedFiles([]);
      await loadDocuments();
      await loadAdminStatus();
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="space-y-4">
      {!embedded && (
        <Panel className="p-5 sm:p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <Eyebrow>Knowledge base</Eyebrow>
              <h1 className="mt-4 text-3xl font-semibold tracking-[-0.04em] text-[color:var(--text-primary)]">
                Document operations
              </h1>
              <p className="mt-2 max-w-3xl text-sm leading-7 text-[color:var(--text-secondary)]">
                Upload source files, trigger indexing, and monitor processing state without leaving the unified shell.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant="secondary" onClick={loadDocuments}>
                Refresh
              </Button>
              <Button onClick={handleReindex} disabled={reindexing || adminStatus?.busy}>
                {reindexing ? 'Reindexing...' : 'Reindex'}
              </Button>
            </div>
          </div>
        </Panel>
      )}

      {(message || error) && (
        <div
          className={cn(
            'rounded-[24px] border px-4 py-3 text-sm',
            error
              ? 'border-rose-200/60 bg-rose-500/10 text-rose-700 dark:border-rose-500/20 dark:text-rose-300'
              : 'border-emerald-200/60 bg-emerald-500/10 text-emerald-700 dark:border-emerald-500/20 dark:text-emerald-300'
          )}
        >
          {error || message}
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(340px,0.8fr)]">
        <div className="space-y-4">
          <Panel className="p-5 sm:p-6">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <div className="text-lg font-semibold text-[color:var(--text-primary)]">Upload documents</div>
                <div className="text-sm text-[color:var(--text-secondary)]">PDF, DOCX, and TXT files are supported.</div>
              </div>
              {embedded && (
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" onClick={loadDocuments}>
                    Refresh
                  </Button>
                  <Button onClick={handleReindex} disabled={reindexing || adminStatus?.busy}>
                    {reindexing ? 'Reindexing...' : 'Reindex'}
                  </Button>
                </div>
              )}
            </div>

            <div className="rounded-[24px] border border-dashed border-[color:var(--border-strong)] bg-[color:var(--card-subtle)] p-5">
              <input
                className="block w-full text-sm text-[color:var(--text-secondary)] file:mr-4 file:rounded-2xl file:border-0 file:bg-[color:var(--accent-soft)] file:px-4 file:py-3 file:font-semibold file:text-[color:var(--accent)]"
                type="file"
                multiple
                accept=".pdf,.docx,.txt"
                onChange={(event) => setSelectedFiles(Array.from(event.target.files || []))}
              />
              {selectedFiles.length > 0 && (
                <div className="mt-4 rounded-[20px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] p-4 text-sm text-[color:var(--text-secondary)]">
                  {selectedFiles.map((file) => file.name).join(', ')}
                </div>
              )}
              <Button className="mt-4 w-full" onClick={handleUpload} disabled={uploading || adminStatus?.busy}>
                {uploading ? 'Uploading...' : 'Upload and refresh'}
              </Button>
            </div>
          </Panel>

          <Panel className="p-5 sm:p-6">
            <div className="mb-4 text-lg font-semibold text-[color:var(--text-primary)]">Documents</div>
            {loading ? (
              <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] px-4 py-5 text-sm text-[color:var(--text-secondary)]">
                Loading documents...
              </div>
            ) : documents.length === 0 ? (
              <div className="rounded-[22px] border border-dashed border-[color:var(--border-soft)] px-4 py-8 text-center text-sm text-[color:var(--text-tertiary)]">
                No documents found.
              </div>
            ) : (
              <div className="space-y-3">
                {documents.map((document) => (
                  <div
                    key={document.name}
                    className="flex items-center gap-4 rounded-[24px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-4"
                  >
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-[18px] bg-[color:var(--accent-soft)] text-xs font-bold uppercase tracking-[0.16em] text-[color:var(--accent)]">
                      {document.type}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-semibold text-[color:var(--text-primary)]">{document.name}</div>
                      <div className="mt-1 text-xs text-[color:var(--text-tertiary)]">
                        {document.size_label} - Updated {formatUpdatedAt(document.updated_at)}
                      </div>
                    </div>
                    <Pill className="border-transparent bg-emerald-500/10 text-emerald-700 dark:text-emerald-300">
                      Ready
                    </Pill>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>

        <Panel className="p-5 sm:p-6">
          <div className="mb-4 text-lg font-semibold text-[color:var(--text-primary)]">Processing status</div>
          <div className="grid gap-3 sm:grid-cols-2">
            <StatusCard label="State" value={adminStatus?.busy ? 'Updating' : 'Ready'} />
            <StatusCard label="Stage" value={adminStatus?.stage || 'Idle'} />
            <StatusCard label="Chunks" value={String(adminStatus?.last_result?.chunk_count ?? '-')} />
            <StatusCard label="OCR pages" value={String(adminStatus?.last_result?.ocr_pages ?? '-')} />
          </div>
          <div className="mt-4 rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-4 text-sm leading-7 text-[color:var(--text-secondary)]">
            While processing is active, chat requests pause automatically so retrieval stays consistent.
          </div>
        </Panel>
      </div>
    </div>
  );
}

function EmptyChatState({ onPromptSelect, festive = false }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="max-w-3xl">
        <div className="text-center">
          <Eyebrow>Start here</Eyebrow>
          <h2 className="mt-5 text-4xl font-semibold tracking-[-0.05em] text-[color:var(--text-primary)]">
            Ask a procurement question.
          </h2>
          <p className="mt-4 text-base leading-8 text-[color:var(--text-secondary)]">
            Use plain language, request a table, or ask for a concise summary. The interface stays intentionally quiet so the answer can do the work.
          </p>
        </div>

        <div className="mt-8 grid gap-3 md:grid-cols-3">
          {SAMPLE_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              type="button"
              onClick={() => onPromptSelect(prompt)}
              className={cn(
                'rounded-[24px] border p-4 text-left transition duration-200',
                festive
                  ? 'festive-chat-surface shadow-soft hover:border-white/40 hover:bg-white/70'
                  : 'border-[color:var(--border-soft)] bg-[color:var(--card-strong)] shadow-soft hover:border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)]'
              )}
            >
              <div className="text-sm font-semibold text-[color:var(--text-primary)]">{prompt}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message, futuristic, festive = false }) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('relative z-20 flex gap-3', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-[18px] bg-gradient-to-br from-primary-500 to-aura-cyan text-xs font-bold text-white shadow-glow">
          PB
        </div>
      )}

      <div
        className={cn(
          'max-w-[86%] rounded-[26px] px-5 py-4 text-sm leading-7',
          festive
            ? isUser
              ? 'festive-message festive-message-user text-white'
              : 'festive-message festive-message-bot text-[color:var(--text-primary)]'
            : isUser
              ? 'border border-transparent bg-[color:var(--accent-soft)] text-[color:var(--text-primary)]'
              : 'border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-primary)]',
          futuristic && !isUser && 'shadow-glow'
        )}
      >
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
          {isUser ? 'You' : 'ProcureBuddy'}
        </div>
        <div className="prose-custom max-w-none text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
      </div>

      {isUser && (
        <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-[18px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-xs font-semibold text-[color:var(--text-secondary)]">
          You
        </div>
      )}
    </div>
  );
}

function ChoiceCard({ active, title, description, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded-[24px] border p-4 text-left transition duration-200',
        active
          ? 'border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]'
          : 'border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-primary)] hover:border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)]'
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-semibold">{title}</div>
        {active && <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--accent)]" />}
      </div>
      <div className={cn('mt-2 text-sm leading-7', active ? 'text-[color:var(--accent)]' : 'text-[color:var(--text-secondary)]')}>
        {description}
      </div>
    </button>
  );
}

function AccentCard({ active, title, onClick, stops }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded-[24px] border p-4 text-left transition duration-200',
        active
          ? 'border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]'
          : 'border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-primary)] hover:border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)]'
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-semibold">{title}</div>
        {active && <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--accent)]" />}
      </div>
      <div
        className="mt-3 h-12 rounded-[18px] border border-white/30 shadow-soft"
        style={{ backgroundImage: `linear-gradient(135deg, ${stops[0]}, ${stops[1]}, ${stops[2]})` }}
      />
    </button>
  );
}

function SettingsLauncherCard({ title, description, icon, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-start gap-4 rounded-[24px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] p-4 text-left transition duration-200 hover:border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)]"
    >
      <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[18px] border border-[color:var(--border-soft)] bg-[color:var(--accent-soft)] text-[color:var(--accent)]">
        {icon}
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-[color:var(--text-primary)]">{title}</span>
        <span className="mt-1 block text-sm leading-6 text-[color:var(--text-secondary)]">{description}</span>
      </span>
    </button>
  );
}

function CloseIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M5 5L15 15M15 5L5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function BackIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
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

function PaletteIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 3.333C6.318 3.333 3.333 6.169 3.333 9.667C3.333 12.703 5.585 15.201 8.542 15.84C9.117 15.965 9.667 15.498 9.667 14.91V14.167C9.667 13.477 10.227 12.917 10.917 12.917H12.083C14.523 12.917 16.5 10.94 16.5 8.5C16.5 5.646 13.987 3.333 10 3.333Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <circle cx="6.5" cy="9" r="0.9" fill="currentColor" />
      <circle cx="8.8" cy="6.8" r="0.9" fill="currentColor" />
      <circle cx="12" cy="6.7" r="0.9" fill="currentColor" />
    </svg>
  );
}

function SparklesIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 3.333L11.607 7.226L15.5 8.833L11.607 10.44L10 14.333L8.393 10.44L4.5 8.833L8.393 7.226L10 3.333Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path d="M15.833 3.75L16.25 4.75L17.25 5.167L16.25 5.583L15.833 6.583L15.417 5.583L14.417 5.167L15.417 4.75L15.833 3.75Z" fill="currentColor" />
      <path d="M4.167 13.417L4.583 14.417L5.583 14.833L4.583 15.25L4.167 16.25L3.75 15.25L2.75 14.833L3.75 14.417L4.167 13.417Z" fill="currentColor" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 3.333L15 5.417V9.375C15 12.407 12.948 15.249 10 16.25C7.052 15.249 5 12.407 5 9.375V5.417L10 3.333Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path d="M8.333 9.583L9.583 10.833L12.083 8.333" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function DatabaseIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 4.167C13.222 4.167 15.833 5.287 15.833 6.667C15.833 8.047 13.222 9.167 10 9.167C6.778 9.167 4.167 8.047 4.167 6.667C4.167 5.287 6.778 4.167 10 4.167ZM4.167 10C4.167 11.38 6.778 12.5 10 12.5C13.222 12.5 15.833 11.38 15.833 10M4.167 13.333C4.167 14.713 6.778 15.833 10 15.833C13.222 15.833 15.833 14.713 15.833 13.333V6.667"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PasswordChecklist({ password }) {
  const checks = getPasswordChecks(password);
  return (
    <div className="mt-3 grid gap-2">
      {PASSWORD_REQUIREMENTS.map((requirement) => (
        <div
          key={requirement.key}
          className={cn(
            'flex items-center gap-2 rounded-2xl border px-3 py-2 text-xs',
            checks[requirement.key]
              ? 'border-emerald-300/60 bg-emerald-500/10 text-emerald-700 dark:border-emerald-500/20 dark:text-emerald-300'
              : 'border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] text-[color:var(--text-secondary)]'
          )}
        >
          <span
            className={cn(
              'h-2 w-2 rounded-full',
              checks[requirement.key] ? 'bg-emerald-500' : 'bg-[color:var(--border-strong)]'
            )}
          />
          {requirement.label}
        </div>
      ))}
    </div>
  );
}

function StatusCard({ label, value }) {
  return (
    <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-4">
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
        {label}
      </div>
      <div className="mt-2 text-lg font-semibold text-[color:var(--text-primary)]">{value}</div>
    </div>
  );
}

function ArrowUpIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 15V5M10 5L6.667 8.333M10 5L13.333 8.333"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
