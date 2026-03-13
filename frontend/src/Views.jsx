import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import JellySlider from './JellySlider';
import JellySwitch from './JellySwitch';
import { api } from './api';

const PASSWORD_REQUIREMENTS = [
  { key: 'length', label: 'At least 8 characters' },
  { key: 'uppercase', label: 'One uppercase letter' },
  { key: 'lowercase', label: 'One lowercase letter' },
  { key: 'digit', label: 'One number' },
  { key: 'symbol', label: 'One special symbol' },
];

const SWATCHES = [
  { c: '#b24b7d', n: 'Rose' },
  { c: '#9d76c8', n: 'Orchid' },
  { c: '#d78a57', n: 'Marigold' },
  { c: '#d96c8d', n: 'Peony' },
  { c: '#7b9b5e', n: 'Leaf' },
  { c: '#7e5a88', n: 'Iris' },
];

function formatUpdatedAt(value) {
  if (!value) {
    return '-';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '-';
  }

  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function statusTone(error) {
  return error ? 'error' : 'info';
}

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
    <div className="password-checklist compact">
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

export function ChatView({ title, messages, loading, sending, error, onSend, onNewChat }) {
  const [input, setInput] = useState('');
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  function submit() {
    const text = input.trim();
    if (!text) {
      return;
    }
    setInput('');
    onSend(text);
  }

  return (
    <>
      <div className="topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: 'var(--success)',
              boxShadow: '0 0 0 3px rgba(34, 197, 94, 0.18)',
            }}
          />
          <span className="topbar-title">{title || 'New Chat'}</span>
          <span className="topbar-chip">Procurement Copilot</span>
        </div>
        <div className="topbar-actions">
          <button className="tb-btn" onClick={onNewChat}>Fresh thread</button>
        </div>
      </div>

      {error && <div className={`status-banner ${statusTone(error)}`}>{error}</div>}

      <div className="chat-area">
        <div className="chat-stage">
          <div className="chat-stage-card">
            <div className="chat-stage-kicker">Live knowledge workspace</div>
            <div className="chat-stage-title">Better answers, cleaner citations, calmer UI.</div>
            <div className="chat-stage-copy">
              Search procurement rules, compare clauses, and keep one thread per decision trail.
            </div>
          </div>
          <div className="chat-stage-orb chat-stage-orb-a" />
          <div className="chat-stage-orb chat-stage-orb-b" />
        </div>

        {loading && <div className="empty-card">Loading chat history...</div>}

        {!loading && messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-title">Ask a procurement question</div>
            <div className="chat-empty-copy">
              Try a slab question, policy clarification, or a table request.
            </div>
            <div className="empty-samples">
              <button className="tb-btn" onClick={() => setInput('8 lakh ka purchase process kya hoga?')}>
                8 lakh process
              </button>
              <button className="tb-btn" onClick={() => setInput('Single tender kab allowed hota hai?')}>
                Single tender rule
              </button>
              <button className="tb-btn" onClick={() => setInput('Show table of procurement process')}>
                Procurement table
              </button>
            </div>
            <div className="chat-empty-panels">
              <div className="chat-empty-panel">
                <span className="chat-empty-panel-kicker">Compare</span>
                OM vs GFR
              </div>
              <div className="chat-empty-panel">
                <span className="chat-empty-panel-kicker">Summarize</span>
                Clause in plain Hindi
              </div>
              <div className="chat-empty-panel">
                <span className="chat-empty-panel-kicker">Extract</span>
                Table + thresholds
              </div>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`msg-row ${message.role}`}>
            <div className={`msg-av ${message.role}`}>
              {message.role === 'assistant' ? 'PB' : 'U'}
            </div>
            <div className={`msg-bubble ${message.role}`}>
              <div className="message-meta">{message.role === 'assistant' ? 'ProcureBuddy' : 'You'}</div>
              <div className="prose-custom">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        ))}

        {sending && (
          <div className="msg-row assistant">
            <div className="msg-av assistant">PB</div>
            <div className="msg-bubble assistant">
              <div className="message-meta">ProcureBuddy</div>
              <div style={{ display: 'flex', gap: 5, padding: '2px 0', alignItems: 'center' }}>
                <div className="t-dot" />
                <div className="t-dot" />
                <div className="t-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <div className="input-wrap">
          <textarea
            className="chat-ta"
            value={input}
            rows={1}
            placeholder="Ask about procurement rules, committees, approvals, or process..."
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
          />
          <button className="send-btn" onClick={submit} disabled={sending}>
            Send
          </button>
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', marginTop: 6 }}>
          Answers are grounded in the loaded procurement documents. Please verify before taking action.
        </div>
      </div>
    </>
  );
}

export function SettingsView({
  accent,
  setAccent,
  theme,
  setTheme,
  festiveMode,
  setFestiveMode,
  activeFestival,
  session,
  onSessionUpdate,
}) {
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [statusError, setStatusError] = useState('');
  const [totpSecret, setTotpSecret] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [qrBase64, setQrBase64] = useState('');
  const [notificationVolume, setNotificationVolume] = useState(0.65);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

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
      } catch (error) {
        if (!cancelled) {
          setStatusError(error.message);
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

  async function handlePasswordChange() {
    if (!password.trim()) {
      setStatusError('Enter a new password first.');
      return;
    }

    if (!isStrongPassword(password)) {
      setStatusError('Use a stronger password with uppercase, lowercase, number, and special symbol.');
      setStatusMessage('');
      return;
    }

    setSubmitting(true);
    try {
      const data = await api.changePassword({
        email: session.email,
        new_password: password.trim(),
      });
      setPassword('');
      setStatusMessage(data.message);
      setStatusError('');
      const freshStatus = await api.getAuthStatus(session.email);
      setStatus(freshStatus);
    } catch (error) {
      setStatusError(error.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleTotpSetup() {
    setSubmitting(true);
    try {
      const data = await api.setupTotp({ email: session.email });
      setTotpSecret(data.secret);
      setQrBase64(data.qr_base64);
      setStatusMessage('Scan the QR code and enter the 6-digit code to enable 2FA.');
      setStatusError('');
    } catch (error) {
      setStatusError(error.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleTotpEnable() {
    if (!totpSecret || !totpCode.trim()) {
      setStatusError('Scan the QR code and enter the current authenticator code.');
      return;
    }

    setSubmitting(true);
    try {
      const data = await api.enableTotp({
        email: session.email,
        secret: totpSecret,
        code: totpCode.trim(),
      });
      const freshStatus = await api.getAuthStatus(session.email);
      setStatus(freshStatus);
      onSessionUpdate({ totpEnabled: true });
      setTotpSecret('');
      setTotpCode('');
      setQrBase64('');
      setStatusMessage(data.message);
      setStatusError('');
    } catch (error) {
      setStatusError(error.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleTotpDisable() {
    setSubmitting(true);
    try {
      const data = await api.disableTotp({ email: session.email });
      const freshStatus = await api.getAuthStatus(session.email);
      setStatus(freshStatus);
      onSessionUpdate({ totpEnabled: false });
      setTotpSecret('');
      setTotpCode('');
      setQrBase64('');
      setStatusMessage(data.message);
      setStatusError('');
    } catch (error) {
      setStatusError(error.message);
      setStatusMessage('');
    } finally {
      setSubmitting(false);
    }
  }

  const activeAccent = useMemo(
    () => SWATCHES.find((swatch) => swatch.c === accent)?.n || accent,
    [accent]
  );

  return (
    <div className="settings-page">
      <div className="settings-title">Settings</div>
      <div className="settings-sub">Theme, security, and account preferences for {session.email}</div>

      {(statusMessage || statusError) && (
        <div className={`status-banner ${statusError ? 'error' : 'success'}`}>
          {statusError || statusMessage}
        </div>
      )}

      <div className="settings-sec">
        <div className="sec-head">Appearance</div>
        <div className="settings-row">
          <div className="row-info">
            <div className="row-label">Theme</div>
            <div className="row-desc">Switch between light, dark, or system theme.</div>
          </div>
          <div className="row-action">
            <select
              value={theme}
              onChange={(event) => setTheme(event.target.value)}
              className="inline-select"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>
        </div>
        <div className="settings-row">
          <div className="row-info">
            <div className="row-label">Accent color</div>
            <div className="row-desc">Current accent: {activeAccent}</div>
          </div>
          <div className="row-action">
            <div className="swatches">
              {SWATCHES.map((swatch) => (
                <div
                  key={swatch.c}
                  className={`swatch${accent === swatch.c ? ' on' : ''}`}
                  style={{ background: swatch.c, '--sw-color': swatch.c }}
                  title={swatch.n}
                  onClick={() => setAccent(swatch.c)}
                />
              ))}
            </div>
          </div>
        </div>
        <div className="settings-row">
          <div className="row-info">
            <div className="row-label">Festive theme engine</div>
            <div className="row-desc">
              Auto-switch decorative themes for major seasonal moments. Current: {activeFestival?.name || 'Default theme'}
            </div>
          </div>
          <div className="row-action">
            <select
              value={festiveMode}
              onChange={(event) => setFestiveMode(event.target.value)}
              className="inline-select"
            >
              <option value="auto">Auto</option>
              <option value="off">Off</option>
            </select>
          </div>
        </div>
      </div>

      <div className="settings-sec">
        <div className="sec-head">Security</div>
        <div className="settings-row">
          <div className="row-info">
            <div className="row-label">Account status</div>
            <div className="row-desc">
              {loadingStatus
                ? 'Loading account status...'
                : `Created ${formatUpdatedAt(status?.created_at)} | 2FA ${status?.totp_enabled ? 'enabled' : 'disabled'}`}
            </div>
          </div>
          <div className="row-action">
            <span className="tag" style={{ background: 'var(--accent-light)', color: 'var(--accent)' }}>
              {status?.must_change ? 'Password reset required' : 'Ready'}
            </span>
          </div>
        </div>
        <div className="settings-row form-row">
          <div className="row-info">
            <div className="row-label">Change password</div>
            <div className="row-desc">Update your password for future logins.</div>
          </div>
          <div className="row-action form-inline">
            <input
              className="input-field compact"
              type={showPassword ? 'text' : 'password'}
              value={password}
              placeholder="Enter new password"
              onChange={(event) => setPassword(event.target.value)}
            />
            <button className="tb-btn" onClick={() => setShowPassword((value) => !value)} type="button">
              {showPassword ? 'Hide' : 'Show'}
            </button>
            <button className="tb-btn accent" onClick={handlePasswordChange} disabled={submitting}>
              Save
            </button>
          </div>
          <PasswordChecklist password={password} />
        </div>
        <div className="settings-row form-row">
          <div className="row-info">
            <div className="row-label">Two-factor authentication</div>
            <div className="row-desc">
              Optional extra security only. Keep it off if you do not want MFA on every sign-in.
            </div>
          </div>
          <div className="row-action form-stack">
            <div className="mfa-toggle-row">
              <JellySwitch
                checked={Boolean(status?.totp_enabled)}
                onChange={(nextValue) => {
                  if (nextValue) {
                    handleTotpSetup();
                  } else {
                    handleTotpDisable();
                  }
                }}
                color={accent}
                size={0.92}
              />
              <div className="field-hint" style={{ marginBottom: 0 }}>
                {status?.totp_enabled ? 'Enabled for this account.' : 'Disabled by default.'}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button className="tb-btn" onClick={handleTotpSetup} disabled={submitting}>
                {status?.totp_enabled ? 'Reset 2FA setup' : 'Set up 2FA'}
              </button>
              <button className="tb-btn" onClick={handleTotpDisable} disabled={submitting || !status?.totp_enabled}>
                Disable 2FA
              </button>
            </div>
            {qrBase64 && (
              <div className="qr-box">
                <img src={`data:image/png;base64,${qrBase64}`} alt="TOTP QR code" />
                <div className="field-hint">Secret: {totpSecret}</div>
                <div className="form-inline">
                  <input
                    className="input-field compact"
                    type="text"
                    inputMode="numeric"
                    value={totpCode}
                    placeholder="Enter 6-digit code"
                    onChange={(event) => setTotpCode(event.target.value)}
                  />
                  <button className="tb-btn accent" onClick={handleTotpEnable} disabled={submitting}>
                    Enable
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="settings-sec">
        <div className="sec-head">Preferences</div>
        <div className="settings-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 10 }}>
          <div className="row-label">Notification volume</div>
          <JellySlider
            value={notificationVolume}
            onChange={setNotificationVolume}
            color={accent}
            width={270}
            height={50}
          />
          <div className="field-hint">Local UI preference only. It does not affect backend behavior.</div>
        </div>
      </div>
    </div>
  );
}

export function AdminView({ sessionEmail }) {
  const [documents, setDocuments] = useState([]);
  const [adminStatus, setAdminStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  async function loadDocuments() {
    setLoading(true);
    try {
      const data = await api.listDocuments(sessionEmail);
      setDocuments(data.documents || []);
      setError('');
    } catch (requestError) {
      setError(requestError.message);
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }

  async function loadAdminStatus() {
    try {
      const data = await api.getAdminStatus(sessionEmail);
      setAdminStatus(data);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  useEffect(() => {
    if (!sessionEmail) {
      return undefined;
    }

    loadDocuments();
    loadAdminStatus();

    const poller = window.setInterval(() => {
      loadAdminStatus();
    }, 5000);

    return () => window.clearInterval(poller);
  }, [sessionEmail]);

  async function handleReindex() {
    setReindexing(true);
    try {
      const data = await api.reindexDocuments(sessionEmail);
      setMessage(data.message);
      setError('');
      await loadDocuments();
      await loadAdminStatus();
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
    } finally {
      setReindexing(false);
    }
  }

  async function handleUpload() {
    if (!selectedFiles.length) {
      setError('Choose at least one document first.');
      setMessage('');
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append('files', file));
      const data = await api.uploadDocuments(sessionEmail, formData);
      setMessage(data.message);
      setError('');
      setSelectedFiles([]);
      await loadDocuments();
      await loadAdminStatus();
    } catch (requestError) {
      setError(requestError.message);
      setMessage('');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="admin-page">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }}>Knowledge Base</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>
            Upload documents, trigger OCR/chunk refresh, and monitor processing in one place.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 7 }}>
          <button className="tb-btn" onClick={loadDocuments}>Refresh</button>
          <button className="tb-btn accent" onClick={handleReindex} disabled={reindexing || adminStatus?.busy}>
            {reindexing ? 'Reindexing...' : 'Reindex'}
          </button>
        </div>
      </div>

      {(message || error) && (
        <div className={`status-banner ${error ? 'error' : 'success'}`}>{error || message}</div>
      )}

      <div className="empty-card admin-status-card" style={{ marginBottom: 16 }}>
        <div className="row-label">Processing control</div>
        <div className="row-desc" style={{ marginBottom: 12 }}>
          While document processing is running, chat requests pause automatically so OCR, rechunking, and vector refresh can finish cleanly.
        </div>
        <div className="admin-status-grid">
          <div className="admin-status-pill">
            State: {adminStatus?.busy ? 'Updating knowledge base' : 'Ready'}
          </div>
          <div className="admin-status-pill">
            Stage: {adminStatus?.stage || 'idle'}
          </div>
          <div className="admin-status-pill">
            Last chunks: {adminStatus?.last_result?.chunk_count ?? '-'}
          </div>
          <div className="admin-status-pill">
            OCR pages: {adminStatus?.last_result?.ocr_pages ?? '-'}
          </div>
        </div>
      </div>

      <div className="settings-sec" style={{ marginBottom: 16 }}>
        <div className="sec-head">Upload documents</div>
        <div className="settings-row form-row">
          <div className="row-info">
            <div className="row-label">Supported formats</div>
            <div className="row-desc">PDF, DOCX, and TXT. Uploading automatically refreshes the knowledge base.</div>
          </div>
          <div className="row-action form-stack">
            <input
              className="input-field"
              type="file"
              multiple
              accept=".pdf,.docx,.txt"
              onChange={(event) => setSelectedFiles(Array.from(event.target.files || []))}
            />
            {selectedFiles.length > 0 && (
              <div className="field-hint">
                Selected: {selectedFiles.map((file) => file.name).join(', ')}
              </div>
            )}
            <button className="tb-btn accent" onClick={handleUpload} disabled={uploading || adminStatus?.busy}>
              {uploading ? 'Uploading and processing...' : 'Upload and refresh KB'}
            </button>
          </div>
        </div>
      </div>

      <div className="empty-card" style={{ marginBottom: 16 }}>
        New uploads are saved to the server `data/` directory, then the app re-runs extraction, OCR fallback for scanned PDFs, rechunking, and vector indexing automatically.
      </div>

      {loading && <div className="empty-card">Loading documents...</div>}
      {!loading && documents.length === 0 && <div className="empty-card">No knowledge-base documents found.</div>}

      {documents.map((doc) => (
        <div key={doc.name} className="doc-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
            <div className="doc-icon">{doc.type.toUpperCase()}</div>
            <div>
              <div style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>{doc.name}</div>
              <div className="doc-meta">{doc.size_label} | Updated {formatUpdatedAt(doc.updated_at)}</div>
            </div>
            <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
              <span className="tag" style={{ background: 'rgba(34, 197, 94, 0.1)', color: '#16a34a' }}>
                Ready
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
