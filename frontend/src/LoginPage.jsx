import { useMemo, useState } from 'react';
import HumanVerificationSlider from './HumanVerificationSlider';
import { api } from './api';

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
    subtitle: 'Sign in with your ProcureBuddy account to continue.',
  },
  register: {
    title: 'Create your account',
    subtitle: 'Start registration with your official email address.',
  },
  verify: {
    title: 'Verify registration',
    subtitle: 'Enter the OTP sent to your inbox and choose a password.',
  },
  totp: {
    title: 'Two-factor verification',
    subtitle: 'Enter the current authenticator code to finish sign in.',
  },
  password: {
    title: 'Change your password',
    subtitle: 'Your temporary password must be replaced before continuing.',
  },
  reset: {
    title: 'Reset password',
    subtitle: 'Generate a temporary password for an existing account.',
  },
};

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

function PasswordChecklist({ password }) {
  const checks = getPasswordChecks(password);

  return (
    <div className="password-checklist">
      {PASSWORD_REQUIREMENTS.map((requirement) => (
        <div
          key={requirement.key}
          className={`password-check${checks[requirement.key] ? ' on' : ''}`}
        >
          <span className="password-check-icon" aria-hidden="true" />
          {requirement.label}
        </div>
      ))}
    </div>
  );
}

function ConfirmPasswordHint({ password, confirmPassword }) {
  if (!confirmPassword) {
    return null;
  }

  const matches = password === confirmPassword;

  return (
    <div className={`password-match-hint${matches ? ' matched' : ' unmatched'}`}>
      {matches ? 'Matched' : 'Not matched'}
    </div>
  );
}

export default function LoginPage({ onAuthenticated }) {
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [otp, setOtp] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [verified, setVerified] = useState(false);
  const [verifKey, setVerifKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const [pendingEmail, setPendingEmail] = useState('');
  const [showLoginPassword, setShowLoginPassword] = useState(false);
  const [showCreatePassword, setShowCreatePassword] = useState(false);
  const [showCreateConfirmPassword, setShowCreateConfirmPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showNewConfirmPassword, setShowNewConfirmPassword] = useState(false);

  const content = useMemo(() => MODE_CONTENT[mode], [mode]);
  const verificationEnabled = useMemo(() => {
    const trimmedEmail = email.trim();
    const trimmedOtp = otp.trim();
    const trimmedTotp = totpCode.trim();
    const passwordReady = password.trim().length > 0;
    const confirmReady = confirmPassword.trim().length > 0;

    switch (mode) {
      case 'login':
        return Boolean(trimmedEmail && passwordReady);
      case 'register':
      case 'reset':
        return Boolean(trimmedEmail);
      case 'verify':
        return Boolean((pendingEmail || trimmedEmail) && trimmedOtp && passwordReady && confirmReady);
      case 'totp':
        return Boolean((pendingEmail || trimmedEmail) && trimmedTotp);
      case 'password':
        return Boolean((pendingEmail || trimmedEmail) && passwordReady && confirmReady);
      default:
        return false;
    }
  }, [mode, email, password, confirmPassword, otp, totpCode, pendingEmail]);
  const verificationHint = useMemo(() => {
    switch (mode) {
      case 'login':
        return 'Enter your email and password, then slide to verify.';
      case 'register':
        return 'Enter your official email address first.';
      case 'verify':
        return 'Enter the OTP and both password fields first.';
      case 'totp':
        return 'Enter your authenticator code first.';
      case 'password':
        return 'Enter and confirm the new password first.';
      case 'reset':
        return 'Enter your email address first.';
      default:
        return 'Complete the required fields first.';
    }
  }, [mode]);

  function resetVerification() {
    setVerified(false);
    setVerifKey((value) => value + 1);
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
    setShowLoginPassword(false);
    setShowCreatePassword(false);
    setShowCreateConfirmPassword(false);
    setShowNewPassword(false);
    setShowNewConfirmPassword(false);
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
    if (!requireVerification()) {
      return;
    }

    setLoading(true);
    try {
      const data = await api.login({ email: email.trim(), password });
      setError('');
      setMessage('');

      if (data.must_change) {
        setPendingEmail(data.email);
        switchMode('password');
        setMessage('Password reset required before access is granted.');
        return;
      }

      if (data.totp_required) {
        setPendingEmail(data.email);
        switchMode('totp');
        setMessage('Enter the authenticator code to complete sign in.');
        return;
      }

      onAuthenticated({
        email: data.email,
        totpEnabled: data.totp_enabled,
      });
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
      resetVerification();
    } finally {
      setLoading(false);
    }
  }

  async function handleRegisterStart() {
    if (!requireVerification()) {
      return;
    }

    setLoading(true);
    try {
      const trimmedEmail = email.trim();
      const data = await api.registerStart({ email: trimmedEmail });
      setPendingEmail(trimmedEmail);
      switchMode('verify');
      setError('');
      setMessage(data.message);
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
      resetVerification();
    } finally {
      setLoading(false);
    }
  }

  async function handleRegisterVerify() {
    if (!requireVerification()) {
      return;
    }

    if (!password.trim() || password !== confirmPassword) {
      setError('Passwords must match before verification.');
      return;
    }

    if (!isStrongPassword(password)) {
      setError('Create a stronger password before continuing.');
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
      setError('');
      setMessage(`${data.message} You can sign in now.`);
      setPassword('');
      setConfirmPassword('');
      setOtp('');
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
      resetVerification();
    } finally {
      setLoading(false);
    }
  }

  async function handleTotpVerify() {
    if (!requireVerification()) {
      return;
    }

    setLoading(true);
    try {
      await api.verifyTotp({ email: pendingEmail || email.trim(), code: totpCode.trim() });
      const status = await api.getAuthStatus(pendingEmail || email.trim());
      onAuthenticated({
        email: status.email,
        totpEnabled: status.totp_enabled,
      });
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
      resetVerification();
    } finally {
      setLoading(false);
    }
  }

  async function handleChangePassword() {
    if (!requireVerification()) {
      return;
    }

    if (!password.trim() || password !== confirmPassword) {
      setError('Passwords must match before saving.');
      return;
    }

    if (!isStrongPassword(password)) {
      setError('Create a stronger password before saving.');
      return;
    }

    setLoading(true);
    try {
      const targetEmail = pendingEmail || email.trim();
      const data = await api.changePassword({
        email: targetEmail,
        new_password: password.trim(),
      });
      switchMode('login');
      setEmail(targetEmail);
      setPendingEmail('');
      setPassword('');
      setConfirmPassword('');
      setError('');
      setMessage(`${data.message} Please sign in with the new password.`);
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
      resetVerification();
    } finally {
      setLoading(false);
    }
  }

  async function handleResetPassword() {
    if (!requireVerification()) {
      return;
    }

    setLoading(true);
    try {
      const data = await api.resetPassword({ email: email.trim() });
      switchMode('login');
      setPassword(data.temp_password);
      setError('');
      setMessage(`Temporary password: ${data.temp_password}`);
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
      resetVerification();
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-bg">
      <div className="login-card">
        <div className="login-logo">
          <div className="logo-mark">PB</div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: '-0.5px' }}>ProcureBuddy</div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>CBRI procurement assistant</div>
          </div>
        </div>

        <div className="auth-tabs">
          <button className={`auth-tab${mode === 'login' ? ' on' : ''}`} onClick={() => switchMode('login')}>Sign in</button>
          <button className={`auth-tab${mode === 'register' ? ' on' : ''}`} onClick={() => switchMode('register')}>Register</button>
          <button className={`auth-tab${mode === 'reset' ? ' on' : ''}`} onClick={() => switchMode('reset')}>Reset</button>
        </div>

        <div className="login-title">{content.title}</div>
        <div className="login-subtitle">{content.subtitle}</div>

        {error && <div className="error-msg">{error}</div>}
        {message && <div className="status-banner success">{message}</div>}

        {(mode === 'login' || mode === 'register' || mode === 'reset') && (
          <div className="input-group">
            <label className="input-label">Email</label>
            <input
              className={`input-field${error ? ' err' : ''}`}
              type="email"
              value={email}
              placeholder="name.cbri@csir.res.in"
              onChange={(event) => {
                updateField(setEmail, event.target.value);
              }}
            />
          </div>
        )}

        {mode === 'login' && (
          <div className="input-group">
            <label className="input-label">Password</label>
            <div className="password-field-wrap">
              <input
                className={`input-field${error ? ' err' : ''}`}
                type={showLoginPassword ? 'text' : 'password'}
                value={password}
                placeholder="Enter your password"
                onChange={(event) => {
                  updateField(setPassword, event.target.value);
                }}
                onKeyDown={(event) => event.key === 'Enter' && handleLogin()}
              />
              <button
                type="button"
                className="password-toggle-btn"
                onClick={() => setShowLoginPassword((value) => !value)}
              >
                {showLoginPassword ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>
        )}

        {mode === 'verify' && (
          <>
            <div className="field-hint">Verifying account for {pendingEmail}</div>
            <div className="input-group">
              <label className="input-label">OTP</label>
              <input
                className={`input-field${error ? ' err' : ''}`}
                type="text"
                value={otp}
                placeholder="Enter 6-digit OTP"
                onChange={(event) => {
                  updateField(setOtp, event.target.value);
                }}
              />
            </div>
            <div className="input-group">
              <label className="input-label">Password</label>
              <div className="password-field-wrap">
                <input
                  className={`input-field${error ? ' err' : ''}`}
                  type={showCreatePassword ? 'text' : 'password'}
                  value={password}
                  placeholder="Choose a password"
                  onChange={(event) => {
                    updateField(setPassword, event.target.value);
                  }}
                />
                <button
                  type="button"
                  className="password-toggle-btn"
                  onClick={() => setShowCreatePassword((value) => !value)}
                >
                  {showCreatePassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>
            <div className="input-group">
              <label className="input-label">Confirm password</label>
              <div className="password-field-wrap">
                <input
                  className={`input-field${error ? ' err' : ''}`}
                  type={showCreateConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  placeholder="Repeat password"
                  onChange={(event) => {
                    updateField(setConfirmPassword, event.target.value);
                  }}
                />
                <button
                  type="button"
                  className="password-toggle-btn"
                  onClick={() => setShowCreateConfirmPassword((value) => !value)}
                >
                  {showCreateConfirmPassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>
            <ConfirmPasswordHint password={password} confirmPassword={confirmPassword} />
            <PasswordChecklist password={password} />
          </>
        )}

        {mode === 'totp' && (
          <>
            <div className="field-hint">Authenticator required for {pendingEmail || email}</div>
            <div className="input-group">
              <label className="input-label">Authenticator code</label>
              <input
                className={`input-field${error ? ' err' : ''}`}
                type="text"
                inputMode="numeric"
                value={totpCode}
                placeholder="Enter current 6-digit code"
                onChange={(event) => {
                  updateField(setTotpCode, event.target.value);
                }}
              />
            </div>
          </>
        )}

        {mode === 'password' && (
          <>
            <div className="field-hint">Updating password for {pendingEmail || email}</div>
            <div className="input-group">
              <label className="input-label">New password</label>
              <div className="password-field-wrap">
                <input
                  className={`input-field${error ? ' err' : ''}`}
                  type={showNewPassword ? 'text' : 'password'}
                  value={password}
                  placeholder="Enter new password"
                  onChange={(event) => {
                    updateField(setPassword, event.target.value);
                  }}
                />
                <button
                  type="button"
                  className="password-toggle-btn"
                  onClick={() => setShowNewPassword((value) => !value)}
                >
                  {showNewPassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>
            <div className="input-group">
              <label className="input-label">Confirm password</label>
              <div className="password-field-wrap">
                <input
                  className={`input-field${error ? ' err' : ''}`}
                  type={showNewConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  placeholder="Repeat new password"
                  onChange={(event) => {
                    updateField(setConfirmPassword, event.target.value);
                  }}
                />
                <button
                  type="button"
                  className="password-toggle-btn"
                  onClick={() => setShowNewConfirmPassword((value) => !value)}
                >
                  {showNewConfirmPassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>
            <ConfirmPasswordHint password={password} confirmPassword={confirmPassword} />
            <PasswordChecklist password={password} />
          </>
        )}

        <div className="verif-box">
          <div className="verif-label">Human verification</div>
          <HumanVerificationSlider
            key={verifKey}
            disabled={!verificationEnabled}
            disabledText={verificationHint}
            onVerified={() => setVerified(true)}
          />
        </div>

        <div className="btn-wrap">
          {mode === 'login' && (
            <button className="btn-login" onClick={handleLogin} disabled={loading}>
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          )}
          {mode === 'register' && (
            <button className="btn-login" onClick={handleRegisterStart} disabled={loading}>
              {loading ? 'Sending OTP...' : 'Send OTP'}
            </button>
          )}
          {mode === 'verify' && (
            <button className="btn-login" onClick={handleRegisterVerify} disabled={loading}>
              {loading ? 'Creating account...' : 'Verify and create account'}
            </button>
          )}
          {mode === 'totp' && (
            <button className="btn-login" onClick={handleTotpVerify} disabled={loading}>
              {loading ? 'Verifying code...' : 'Verify 2FA'}
            </button>
          )}
          {mode === 'password' && (
            <button className="btn-login" onClick={handleChangePassword} disabled={loading}>
              {loading ? 'Saving password...' : 'Save new password'}
            </button>
          )}
          {mode === 'reset' && (
            <button className="btn-login" onClick={handleResetPassword} disabled={loading}>
              {loading ? 'Generating temp password...' : 'Reset password'}
            </button>
          )}
        </div>

        <div className="auth-switch">
          {mode === 'login' && (
            <>
              Need an account? <button onClick={() => switchMode('register')}>Register here</button>
            </>
          )}
          {mode === 'register' && (
            <>
              Already registered? <button onClick={() => switchMode('login')}>Back to sign in</button>
            </>
          )}
          {mode === 'verify' && (
            <>
              Need a fresh OTP? <button onClick={() => switchMode('register')}>Start again</button>
            </>
          )}
          {mode === 'totp' && (
            <>
              Wrong account? <button onClick={() => switchMode('login')}>Back to sign in</button>
            </>
          )}
          {mode === 'password' && (
            <>
              Use a different account? <button onClick={() => switchMode('login')}>Back to sign in</button>
            </>
          )}
          {mode === 'reset' && (
            <>
              Remembered it? <button onClick={() => switchMode('login')}>Back to sign in</button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
