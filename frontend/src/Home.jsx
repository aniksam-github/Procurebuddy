import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button, Eyebrow, IconButton, Panel, Pill } from './components/ui';
import { useTheme } from './context/ThemeContext';

const FEATURES = [
  {
    title: 'Grounded answers',
    description: 'Ask in natural language and get responses anchored to your procurement knowledge base.',
    icon: SearchIcon,
  },
  {
    title: 'Readable tables',
    description: 'Turn dense rules into clean comparison tables directly inside the chat thread.',
    icon: TableIcon,
  },
  {
    title: 'Trusted access',
    description: 'Support for authentication, password flows, and two-factor verification without UI clutter.',
    icon: ShieldIcon,
  },
];

const SIGNALS = [
  'Light-first premium design',
  'Fast, distraction-free chat',
  'Theme and font controls',
  'Production-ready layout system',
];

export default function Home() {
  const navigate = useNavigate();
  const { setTheme, resolvedTheme } = useTheme();

  return (
    <div className="relative min-h-screen overflow-hidden px-4 py-4 sm:px-6">
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] max-w-7xl flex-col">
        <motion.header
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="sticky top-4 z-30"
        >
          <Panel className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3 sm:px-5">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-500 via-primary-500 to-aura-cyan text-sm font-extrabold text-white shadow-glow">
                PB
              </div>
              <div>
                <div className="text-sm font-semibold tracking-tight text-[color:var(--text-primary)] sm:text-base">
                  ProcureBuddy AI
                </div>
                <div className="text-xs text-[color:var(--text-tertiary)]">
                  Futuristic procurement chat for serious workflows
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <IconButton
                onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
                aria-label="Toggle theme"
                title={`Switch to ${resolvedTheme === 'dark' ? 'light' : 'dark'} mode`}
              >
                {resolvedTheme === 'dark' ? <SunIcon /> : <MoonIcon />}
              </IconButton>
              <Button variant="secondary" className="hidden sm:inline-flex" onClick={() => navigate('/chat')}>
                Open app
              </Button>
              <Button onClick={() => navigate('/chat')}>Start Chatting</Button>
            </div>
          </Panel>
        </motion.header>

        <main className="relative z-10 flex flex-1 flex-col justify-center py-10 sm:py-14">
          <section className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.05 }}
              className="max-w-2xl"
            >
              <Eyebrow className="mb-6">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                Light mode designed first
              </Eyebrow>

              <h1 className="max-w-3xl text-4xl font-semibold leading-[1.02] tracking-[-0.04em] text-[color:var(--text-primary)] sm:text-5xl lg:text-6xl">
                A premium chatbot interface with calm focus and a futuristic edge.
              </h1>

              <p className="mt-6 max-w-xl text-base leading-8 text-[color:var(--text-secondary)] sm:text-lg">
                ProcureBuddy blends clean structure, polished light surfaces, and subtle motion into a
                chat experience built for daily work, not demos.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <Button size="lg" onClick={() => navigate('/chat')}>
                  Start Chatting
                  <ArrowRightIcon />
                </Button>
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={() => document.getElementById('feature-grid')?.scrollIntoView({ behavior: 'smooth' })}
                >
                  Explore the system
                </Button>
              </div>

              <div className="mt-8 flex flex-wrap gap-2.5">
                {SIGNALS.map((item) => (
                  <Pill key={item}>{item}</Pill>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.12 }}
              className="relative"
            >
              <Panel className="overflow-hidden p-4 sm:p-5">
                <div className="rounded-[24px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] p-4 shadow-soft">
                  <div className="flex items-center justify-between border-b border-[color:var(--border-soft)] pb-4">
                    <div>
                      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
                        Interface Preview
                      </div>
                      <div className="mt-1 text-lg font-semibold text-[color:var(--text-primary)]">
                        Chat shell and settings system
                      </div>
                    </div>
                    <Pill className="border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]">
                      Live theme controls
                    </Pill>
                  </div>

                  <div className="mt-4 grid gap-4 md:grid-cols-[220px_1fr]">
                    <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-4">
                      <div className="mb-4 flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-500 to-aura-cyan text-sm font-bold text-white">
                          PB
                        </div>
                        <div>
                          <div className="text-sm font-semibold text-[color:var(--text-primary)]">Sidebar</div>
                          <div className="text-xs text-[color:var(--text-tertiary)]">Collapsible history</div>
                        </div>
                      </div>
                      <div className="space-y-2">
                        {['Vendor comparison', 'Threshold summary', 'Committee workflow'].map((item, index) => (
                          <div
                            key={item}
                            className={`rounded-2xl border px-3 py-3 text-sm ${
                              index === 0
                                ? 'border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]'
                                : 'border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-secondary)]'
                            }`}
                          >
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-4">
                      <div className="mb-4 flex items-center justify-between">
                        <div>
                          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
                            Conversation
                          </div>
                          <div className="mt-1 text-sm font-semibold text-[color:var(--text-primary)]">
                            Minimal, readable, table-capable
                          </div>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="h-2.5 w-2.5 rounded-full bg-rose-300" />
                          <span className="h-2.5 w-2.5 rounded-full bg-amber-300" />
                          <span className="h-2.5 w-2.5 rounded-full bg-emerald-300" />
                        </div>
                      </div>

                      <div className="space-y-3">
                        <MessagePreview
                          role="assistant"
                          label="ProcureBuddy"
                          content="For an 8 lakh purchase, I can summarize the process and render the approval path in a table."
                        />
                        <MessagePreview
                          role="user"
                          label="You"
                          content="Show the comparison between single tender and limited tender."
                        />
                        <div className="rounded-[20px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] p-4">
                          <div className="mb-3 flex items-center justify-between">
                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--text-tertiary)]">
                              Table support
                            </div>
                            <span className="text-xs text-[color:var(--text-tertiary)]">Rendered in chat</span>
                          </div>
                          <div className="overflow-hidden rounded-2xl border border-[color:var(--border-soft)]">
                            <div className="grid grid-cols-2 bg-[color:var(--accent-soft)] text-xs font-semibold uppercase tracking-[0.12em] text-[color:var(--text-tertiary)]">
                              <div className="px-3 py-2">Mode</div>
                              <div className="px-3 py-2">When used</div>
                            </div>
                            <div className="grid grid-cols-2 text-sm text-[color:var(--text-secondary)]">
                              <div className="border-t border-[color:var(--border-soft)] px-3 py-3">Single tender</div>
                              <div className="border-t border-[color:var(--border-soft)] px-3 py-3">
                                Exception-based procurement
                              </div>
                              <div className="border-t border-[color:var(--border-soft)] px-3 py-3">Limited tender</div>
                              <div className="border-t border-[color:var(--border-soft)] px-3 py-3">
                                Restricted vendor shortlist
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Panel>
            </motion.div>
          </section>

          <section id="feature-grid" className="mt-16 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="grid gap-4 sm:grid-cols-3">
              {FEATURES.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: '-40px' }}
                    transition={{ duration: 0.35, delay: index * 0.08 }}
                  >
                    <Panel className="h-full p-5">
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[color:var(--accent-soft)] text-[color:var(--accent)]">
                        <Icon />
                      </div>
                      <div className="mt-5 text-lg font-semibold text-[color:var(--text-primary)]">{feature.title}</div>
                      <p className="mt-2 text-sm leading-7 text-[color:var(--text-secondary)]">
                        {feature.description}
                      </p>
                    </Panel>
                  </motion.div>
                );
              })}
            </div>

            <Panel className="p-6 sm:p-7">
              <Eyebrow>Design system</Eyebrow>
              <h2 className="mt-5 text-3xl font-semibold tracking-[-0.04em] text-[color:var(--text-primary)]">
                One visual language across landing, login, chat, and settings.
              </h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-[color:var(--text-secondary)] sm:text-base">
                Spacing, surfaces, controls, and typography now work as a single system. Light mode stays
                the reference experience, while dark mode remains complete and elegant.
              </p>
              <div className="mt-8 grid gap-3 sm:grid-cols-2">
                <Panel className="p-4">
                  <div className="text-sm font-semibold text-[color:var(--text-primary)]">Light mode first</div>
                  <p className="mt-2 text-sm leading-6 text-[color:var(--text-secondary)]">
                    Bright layered surfaces, soft borders, and premium contrast for long-form reading.
                  </p>
                </Panel>
                <Panel className="p-4">
                  <div className="text-sm font-semibold text-[color:var(--text-primary)]">Subtle motion</div>
                  <p className="mt-2 text-sm leading-6 text-[color:var(--text-secondary)]">
                    Framer Motion adds polish through restrained reveals and hover feedback, never noise.
                  </p>
                </Panel>
              </div>
            </Panel>
          </section>
        </main>
      </div>
    </div>
  );
}

function MessagePreview({ role, label, content }) {
  const isUser = role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[88%] rounded-[22px] px-4 py-3 ${
          isUser
            ? 'border border-transparent bg-[color:var(--accent-soft)] text-[color:var(--text-primary)]'
            : 'border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-primary)]'
        }`}
      >
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
          {label}
        </div>
        <div className="text-sm leading-7 text-[color:var(--text-secondary)]">{content}</div>
      </div>
    </div>
  );
}

function ArrowRightIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M4.167 10H15.833M10.833 5L15.833 10L10.833 15"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M14.583 14.583L17.5 17.5M16.667 9.167A7.5 7.5 0 1 1 1.667 9.167A7.5 7.5 0 0 1 16.667 9.167Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TableIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M3.333 5.833C3.333 4.913 4.08 4.167 5 4.167H15C15.92 4.167 16.667 4.913 16.667 5.833V14.167C16.667 15.087 15.92 15.833 15 15.833H5C4.08 15.833 3.333 15.087 3.333 14.167V5.833ZM3.333 8.333H16.667M8.333 4.167V15.833"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 2.5L15.833 5V9.833C15.833 13.217 13.504 16.24 10 17.5C6.496 16.24 4.167 13.217 4.167 9.833V5L10 2.5ZM7.917 10L9.167 11.25L12.5 7.917"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg className="h-[18px] w-[18px]" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M10 2.5V4.167M10 15.833V17.5M4.697 4.697L5.875 5.875M14.125 14.125L15.303 15.303M2.5 10H4.167M15.833 10H17.5M4.697 15.303L5.875 14.125M14.125 5.875L15.303 4.697M13.333 10A3.333 3.333 0 1 1 6.667 10A3.333 3.333 0 0 1 13.333 10Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg className="h-[18px] w-[18px]" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M12.668 3.115C11.903 2.864 11.086 2.731 10.238 2.731C5.963 2.731 2.497 6.196 2.497 10.472C2.497 14.748 5.963 18.213 10.238 18.213C13.932 18.213 17.022 15.625 17.796 12.164C17.117 12.508 16.348 12.702 15.533 12.702C12.775 12.702 10.539 10.466 10.539 7.709C10.539 5.855 11.547 4.236 12.668 3.115Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
