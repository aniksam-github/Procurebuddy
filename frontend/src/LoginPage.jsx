import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import HumanVerificationSlider from './HumanVerificationSlider';
import { api } from './api';
import { Button, Eyebrow, IconButton, Panel, Pill, cn } from './components/ui';
import { useTheme } from './context/ThemeContext';

const PASSWORD_REQUIREMENTS = [
  { key: 'length', label: 'At least 8 characters' },
  { key: 'uppercase', label: 'One uppercase letter' },
  { key: 'lowercase', label: 'One lowercase letter' },
  { key: 'digit', label: 'One number' },
  { key: 'symbol', label: 'One special symbol' },
];

const MODE_CONTENT = {
  login: {
    title: 'Welcome back',
    subtitle: 'Sign in to your procurement workspace.',
    action: 'Sign in',
  },
  register: {
    title: 'Create account',
    subtitle: 'Start registration with your official email.',
    action: 'Send OTP',
  },
  verify: {
    title: 'Verify registration',
    subtitle: 'Enter the OTP and set your password.',
    action: 'Verify and create account',
  },
  totp: {
    title: 'Two-factor verification',
    subtitle: 'Confirm the login using your authenticator app.',
    action: 'Verify 2FA',
  },
  password: {
    title: 'Update password',
    subtitle: 'Your password needs to be changed before continuing.',
    action: 'Save new password',
  },
  reset: {
    title: 'Reset password',
    subtitle: 'Generate a temporary password for an existing account.',
    action: 'Reset password',
  },
};

const BENEFITS = [
  'Consistent light-first workspace',
  'Minimal chat optimized for readability',
  'Theme, UI mode, and font preferences',
  'Secure login with optional TOTP verification',
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

function buildSession(primary = {}, fallback = {}) {
  return {
    email: primary.email || fallback.email || '',
    displayName: primary.display_name || fallback.display_name || '',
    username: primary.username || fallback.username || '',
    avatarBase64: primary.avatar_base64 || fallback.avatar_base64 || '',
    totpEnabled: primary.totp_enabled ?? fallback.totp_enabled,
    is_admin: primary.is_admin ?? fallback.is_admin,
    token: primary.token || fallback.token || '',
    accessToken: primary.accessToken || fallback.accessToken || '',
    access_token: primary.access_token || fallback.access_token || '',
  };
}

export default function LoginPage({ onAuthenticated }) {
  const navigate = useNavigate();
  const { setTheme, resolvedTheme } = useTheme();
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [otp, setOtp] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [verified, setVerified] = useState(false);
  const [verificationKey, setVerificationKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const [pendingEmail, setPendingEmail] = useState('');
  const [showPassword, setShowPassword] = useState({
    login: false,
    create: false,
    createConfirm: false,
    newPassword: false,
    newConfirm: false,
  });

  const content = MODE_CONTENT[mode];

  const verificationEnabled = useMemo(() => {
    const trimmedEmail = email.trim();
    const trimmedPendingEmail = pendingEmail.trim();
    switch (mode) {
      case 'login':
        return Boolean(trimmedEmail && password.trim());
      case 'register':
      case 'reset':
        return Boolean(trimmedEmail);
      case 'verify':
        return Boolean((trimmedPendingEmail || trimmedEmail) && otp.trim() && password.trim() && confirmPassword.trim());
      case 'totp':
        return Boolean((trimmedPendingEmail || trimmedEmail) && totpCode.trim());
      case 'password':
        return Boolean((trimmedPendingEmail || trimmedEmail) && password.trim() && confirmPassword.trim());
      default:
        return false;
    }
  }, [mode, email, pendingEmail, otp, password, confirmPassword, totpCode]);

  const verificationHint = useMemo(() => {
    switch (mode) {
      case 'login':
        return 'Enter your email and password to enable verification.';
      case 'register':
        return 'Enter your official email to continue.';
      case 'verify':
        return 'Fill OTP and both password fields first.';
      case 'totp':
        return 'Enter your authenticator code first.';
      case 'password':
        return 'Enter and confirm the new password first.';
      case 'reset':
        return 'Enter your account email first.';
      default:
        return 'Complete the required fields first.';
    }
  }, [mode]);

  function resetVerification() {
    setVerified(false);
    setVerificationKey((value) => value + 1);
  }

  function updateField(setter, value) {
    setter(value);
    setError('');
    if (verified) {
      resetVerification();
    }
  }

  function switchMode(nextMode) {
    setMode(nextMode);
    setError('');
    setMessage('');
    setPassword('');
    setConfirmPassword('');
    setOtp('');
    setTotpCode('');
    setShowPassword({
      login: false,
      create: false,
      createConfirm: false,
      newPassword: false,
      newConfirm: false,
    });
    resetVerification();
  }

  function requireVerification() {
    if (!verified) {
      setError('Complete the human verification slider first.');
      return false;
    }
    return true;
  }

  async function handleLogin() {
    if (!requireVerification()) return;
    setLoading(true);
    try {
      const data = await api.login({ email: email.trim(), password });
      setError('');
      setMessage('');

      if (data.must_change) {
        setPendingEmail(data.email);
        switchMode('password');
        setMessage('Password reset required before continuing.');
        return;
      }

      if (data.totp_required) {
        setPendingEmail(data.email);
        switchMode('totp');
        setMessage('Enter the authenticator code to finish sign in.');
        return;
      }

      onAuthenticated(buildSession(data));
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function handleRegisterStart() {
    if (!requireVerification()) return;
    setLoading(true);
    try {
      const nextEmail = email.trim();
      const data = await api.registerStart({ email: nextEmail });
      setPendingEmail(nextEmail);
      switchMode('verify');
      setMessage(data.message || 'OTP sent. Verify to finish account creation.');
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function handleRegisterVerify() {
    if (!requireVerification()) return;
    if (password.trim() !== confirmPassword.trim()) {
      setError('Passwords must match.');
      return;
    }
    if (!isStrongPassword(password.trim())) {
      setError('Choose a stronger password.');
      return;
    }

    setLoading(true);
    try {
      const data = await api.registerVerify({
        email: pendingEmail || email.trim(),
        otp: otp.trim(),
        password: password.trim(),
      });
      switchMode('login');
      setMessage(`${data.message || 'Account created.'} Sign in to continue.`);
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function handleTotpVerify() {
    if (!requireVerification()) return;
    setLoading(true);
    try {
      const targetEmail = pendingEmail || email.trim();
      const verification = await api.verifyTotp({ email: targetEmail, code: totpCode.trim() });
      const status = await api.getAuthStatus(targetEmail);
      onAuthenticated(buildSession(status, verification));
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function handleChangePassword() {
    if (!requireVerification()) return;
    if (password.trim() !== confirmPassword.trim()) {
      setError('Passwords must match.');
      return;
    }
    if (!isStrongPassword(password.trim())) {
      setError('Choose a stronger password.');
      return;
    }

    setLoading(true);
    try {
      const targetEmail = pendingEmail || email.trim();
      const data = await api.changePassword({ email: targetEmail, new_password: password.trim() });
      setEmail(targetEmail);
      setPendingEmail('');
      switchMode('login');
      setMessage(`${data.message || 'Password updated.'} Sign in with your new password.`);
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function handleResetPassword() {
    if (!requireVerification()) return;
    setLoading(true);
    try {
      const data = await api.resetPassword({ email: email.trim() });
      switchMode('login');
      setPassword(data.temp_password || '');
      setMessage(`Temporary password: ${data.temp_password}`);
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  function submitCurrentMode() {
    switch (mode) {
      case 'login':
        return handleLogin();
      case 'register':
        return handleRegisterStart();
      case 'verify':
        return handleRegisterVerify();
      case 'totp':
        return handleTotpVerify();
      case 'password':
        return handleChangePassword();
      default:
        return handleResetPassword();
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden px-4 py-4 sm:px-6">
      <div className="mx-auto flex min-h-[calc(100vh-2rem)] max-w-7xl flex-col">
        <div className="flex items-center justify-between gap-3 pb-4">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="inline-flex items-center gap-3 rounded-2xl border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] px-3 py-2 text-left shadow-[var(--shadow-panel)] transition duration-200 hover:bg-[color:var(--card-hover)]"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-500 to-aura-cyan text-sm font-extrabold text-white">
              PB
            </div>
            <div>
              <div className="text-sm font-semibold text-[color:var(--text-primary)]">ProcureBuddy AI</div>
              <div className="text-xs text-[color:var(--text-tertiary)]">Premium procurement workspace</div>
            </div>
          </button>

          <div className="flex items-center gap-2">
            <IconButton
              onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
              aria-label="Toggle theme"
            >
              {resolvedTheme === 'dark' ? <SunIcon /> : <MoonIcon />}
            </IconButton>
            <Button variant="secondary" onClick={() => navigate('/')}>
              Home
            </Button>
          </div>
        </div>

        <div className="grid flex-1 gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <Panel className="hidden overflow-hidden p-8 lg:flex lg:flex-col lg:justify-between">
            <div>
              <Eyebrow>
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                Secure access
              </Eyebrow>
              <h1 className="mt-6 max-w-xl text-5xl font-semibold leading-[1.02] tracking-[-0.05em] text-[color:var(--text-primary)]">
                Clean entry point, same premium system.
              </h1>
              <p className="mt-5 max-w-xl text-base leading-8 text-[color:var(--text-secondary)]">
                The authentication flow uses the same light-first surfaces, subtle depth, and measured
                motion as the main chat experience.
              </p>
            </div>

            <div className="mt-10 grid gap-4">
              <div className="grid gap-3 sm:grid-cols-2">
                {BENEFITS.map((benefit) => (
                  <Panel key={benefit} className="p-4">
                    <div className="flex items-start gap-3">
                      <span className="mt-1 h-2.5 w-2.5 rounded-full bg-[color:var(--accent)]" />
                      <div className="text-sm leading-7 text-[color:var(--text-secondary)]">{benefit}</div>
                    </div>
                  </Panel>
                ))}
              </div>

              <Panel className="p-5">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--text-tertiary)]">
                  Experience preview
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <PreviewMetric label="Main design focus" value="Light mode" />
                  <PreviewMetric label="UI modes" value="Minimal + Futuristic" />
                  <PreviewMetric label="Default font" value="Inter" />
                  <PreviewMetric label="Responsive shell" value="Sidebar + topbar" />
                </div>
              </Panel>
            </div>
          </Panel>

          <Panel className="flex items-center justify-center p-4 sm:p-6 lg:p-8">
            <div className="w-full max-w-md">
              <div className="mb-6">
                <Eyebrow>Account access</Eyebrow>
                <AnimatePresence mode="wait">
                  <motion.div
                    key={mode}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.18 }}
                    className="mt-4"
                  >
                    <h2 className="text-3xl font-semibold tracking-[-0.04em] text-[color:var(--text-primary)]">
                      {content.title}
                    </h2>
                    <p className="mt-2 text-sm leading-7 text-[color:var(--text-secondary)]">
                      {content.subtitle}
                    </p>
                  </motion.div>
                </AnimatePresence>
              </div>

              <div className="mb-6 grid grid-cols-3 gap-2 rounded-[24px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] p-1.5">
                {['login', 'register', 'reset'].map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => switchMode(tab)}
                    className={cn(
                      'rounded-[18px] px-3 py-2.5 text-sm font-semibold transition duration-200',
                      mode === tab
                        ? 'bg-[color:var(--card-strong)] text-[color:var(--text-primary)] shadow-soft'
                        : 'text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)]'
                    )}
                  >
                    {tab === 'login' ? 'Sign in' : tab === 'register' ? 'Register' : 'Reset'}
                  </button>
                ))}
              </div>

              {error && (
                <div className="mb-4 rounded-2xl border border-rose-200/60 bg-rose-500/10 px-4 py-3 text-sm text-rose-600 dark:border-rose-500/20 dark:text-rose-300">
                  {error}
                </div>
              )}
              {message && (
                <div className="mb-4 rounded-2xl border border-emerald-200/60 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-500/20 dark:text-emerald-300">
                  {message}
                </div>
              )}

              <div className="space-y-4">
                {(mode === 'login' || mode === 'register' || mode === 'reset') && (
                  <Field label="Email">
                    <input
                      className="app-input"
                      type="email"
                      value={email}
                      placeholder="name@example.com"
                      onChange={(event) => updateField(setEmail, event.target.value)}
                    />
                  </Field>
                )}

                {mode === 'login' && (
                  <Field label="Password">
                    <PasswordInput
                      value={password}
                      onChange={(value) => updateField(setPassword, value)}
                      show={showPassword.login}
                      onToggle={() => setShowPassword((current) => ({ ...current, login: !current.login }))}
                      placeholder="Enter your password"
                      onEnter={handleLogin}
                    />
                  </Field>
                )}

                {mode === 'verify' && (
                  <>
                    <Pill className="border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]">
                      Verifying {pendingEmail}
                    </Pill>
                    <Field label="OTP">
                      <input
                        className="app-input"
                        type="text"
                        value={otp}
                        placeholder="Enter the 6-digit OTP"
                        onChange={(event) => updateField(setOtp, event.target.value)}
                      />
                    </Field>
                    <Field label="Password">
                      <PasswordInput
                        value={password}
                        onChange={(value) => updateField(setPassword, value)}
                        show={showPassword.create}
                        onToggle={() => setShowPassword((current) => ({ ...current, create: !current.create }))}
                        placeholder="Choose a password"
                      />
                    </Field>
                    <Field label="Confirm password">
                      <PasswordInput
                        value={confirmPassword}
                        onChange={(value) => updateField(setConfirmPassword, value)}
                        show={showPassword.createConfirm}
                        onToggle={() =>
                          setShowPassword((current) => ({
                            ...current,
                            createConfirm: !current.createConfirm,
                          }))
                        }
                        placeholder="Repeat your password"
                      />
                    </Field>
                    <ConfirmPasswordHint password={password} confirmPassword={confirmPassword} />
                    <PasswordChecklist password={password} />
                  </>
                )}

                {mode === 'totp' && (
                  <>
                    <Pill className="border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]">
                      Authenticator for {pendingEmail || email}
                    </Pill>
                    <Field label="Authenticator code">
                      <input
                        className="app-input"
                        type="text"
                        inputMode="numeric"
                        value={totpCode}
                        placeholder="Enter the 6-digit code"
                        onChange={(event) => updateField(setTotpCode, event.target.value)}
                      />
                    </Field>
                  </>
                )}

                {mode === 'password' && (
                  <>
                    <Pill className="border-transparent bg-[color:var(--accent-soft)] text-[color:var(--accent)]">
                      Updating {pendingEmail || email}
                    </Pill>
                    <Field label="New password">
                      <PasswordInput
                        value={password}
                        onChange={(value) => updateField(setPassword, value)}
                        show={showPassword.newPassword}
                        onToggle={() =>
                          setShowPassword((current) => ({
                            ...current,
                            newPassword: !current.newPassword,
                          }))
                        }
                        placeholder="Enter a new password"
                      />
                    </Field>
                    <Field label="Confirm password">
                      <PasswordInput
                        value={confirmPassword}
                        onChange={(value) => updateField(setConfirmPassword, value)}
                        show={showPassword.newConfirm}
                        onToggle={() =>
                          setShowPassword((current) => ({
                            ...current,
                            newConfirm: !current.newConfirm,
                          }))
                        }
                        placeholder="Repeat the new password"
                      />
                    </Field>
                    <ConfirmPasswordHint password={password} confirmPassword={confirmPassword} />
                    <PasswordChecklist password={password} />
                  </>
                )}

                <Field label="Human verification">
                  <HumanVerificationSlider
                    key={verificationKey}
                    disabled={!verificationEnabled}
                    disabledText={verificationHint}
                    onVerified={() => setVerified(true)}
                  />
                </Field>

                <Button className="w-full" size="lg" onClick={submitCurrentMode} disabled={loading}>
                  {loading ? 'Processing...' : content.action}
                </Button>

                <div className="text-center text-sm text-[color:var(--text-secondary)]">
                  {mode === 'login' && (
                    <>
                      Need an account?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('register')}
                        className="font-semibold text-[color:var(--accent)]"
                      >
                        Register
                      </button>
                    </>
                  )}
                  {mode === 'register' && (
                    <>
                      Already registered?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('login')}
                        className="font-semibold text-[color:var(--accent)]"
                      >
                        Sign in
                      </button>
                    </>
                  )}
                  {mode === 'verify' && (
                    <>
                      Need a new OTP?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('register')}
                        className="font-semibold text-[color:var(--accent)]"
                      >
                        Start again
                      </button>
                    </>
                  )}
                  {mode === 'totp' && (
                    <>
                      Wrong account?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('login')}
                        className="font-semibold text-[color:var(--accent)]"
                      >
                        Sign in
                      </button>
                    </>
                  )}
                  {mode === 'password' && (
                    <>
                      Different account?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('login')}
                        className="font-semibold text-[color:var(--accent)]"
                      >
                        Sign in
                      </button>
                    </>
                  )}
                  {mode === 'reset' && (
                    <>
                      Remembered it?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('login')}
                        className="font-semibold text-[color:var(--accent)]"
                      >
                        Sign in
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <div className="mb-2 text-sm font-medium text-[color:var(--text-primary)]">{label}</div>
      {children}
    </label>
  );
}

function PasswordInput({ value, onChange, show, onToggle, placeholder, onEnter }) {
  return (
    <div className="relative">
      <input
        className="app-input pr-16"
        type={show ? 'text' : 'password'}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && onEnter) {
            onEnter();
          }
        }}
      />
      <button
        type="button"
        onClick={onToggle}
        className="absolute right-3 top-1/2 -translate-y-1/2 text-sm font-medium text-[color:var(--text-tertiary)] transition duration-200 hover:text-[color:var(--text-primary)]"
      >
        {show ? 'Hide' : 'Show'}
      </button>
    </div>
  );
}

function PreviewMetric({ label, value }) {
  return (
    <div className="rounded-[22px] border border-[color:var(--border-soft)] bg-[color:var(--card-subtle)] px-4 py-3">
      <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--text-tertiary)]">
        {label}
      </div>
      <div className="mt-2 text-lg font-semibold text-[color:var(--text-primary)]">{value}</div>
    </div>
  );
}

function PasswordChecklist({ password }) {
  const checks = getPasswordChecks(password);
  return (
    <div className="grid gap-2 sm:grid-cols-2">
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

function ConfirmPasswordHint({ password, confirmPassword }) {
  if (!confirmPassword) return null;
  const matched = password === confirmPassword;
  return (
    <div className={cn('text-sm font-medium', matched ? 'text-emerald-600 dark:text-emerald-300' : 'text-rose-500')}>
      {matched ? 'Passwords match.' : 'Passwords do not match yet.'}
    </div>
  );
}

function SunIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
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
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden="true">
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
