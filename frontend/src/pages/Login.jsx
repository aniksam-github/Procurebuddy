import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'
import { login as apiLogin, registerStart, registerVerify } from '../api/api.js'

// step values
// login       → normal login form
// reg-email   → register: enter email → send OTP
// reg-otp     → register: enter OTP + set password
// reg-done    → success screen

export default function Login() {
  const { login } = useAuth()
  const navigate  = useNavigate()

  const [step,     setStep]     = useState('login')
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [otp,      setOtp]      = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)

  // ── LOGIN ──────────────────────────────────────────────────────────────────
  async function handleLogin(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await apiLogin(email, password)
      login(email)
      navigate('/chat')
    } catch (err) {
      setError(err?.response?.data?.detail || 'Invalid email or password.')
    } finally {
      setLoading(false)
    }
  }

  // ── REGISTER STEP 1: send OTP ──────────────────────────────────────────────
  async function handleRegisterStart(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await registerStart(email)
      setStep('reg-otp')
    } catch (err) {
      setError(err?.response?.data?.detail || 'Could not send OTP. Check your email domain.')
    } finally {
      setLoading(false)
    }
  }

  // ── REGISTER STEP 2: verify OTP + set password ─────────────────────────────
  async function handleRegisterVerify(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await registerVerify(email, otp, password)
      setStep('reg-done')
    } catch (err) {
      setError(err?.response?.data?.detail || 'OTP verification failed.')
    } finally {
      setLoading(false)
    }
  }

  // ── SHARED STYLES ──────────────────────────────────────────────────────────
  const inputCls = "w-full bg-ink-800 border border-ink-600 rounded-lg px-4 py-3 text-ink-100 placeholder-ink-500 text-sm focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 transition-colors"
  const labelCls = "block text-xs font-mono tracking-widest uppercase text-ink-400 mb-2"
  const btnCls   = "w-full bg-amber-500 hover:bg-amber-400 disabled:bg-ink-600 disabled:cursor-not-allowed text-ink-950 font-semibold rounded-lg py-3 text-sm transition-all duration-150 shadow-[0_0_20px_rgba(245,158,11,0.3)] hover:shadow-[0_0_28px_rgba(245,158,11,0.5)]"

  return (
    <div className="min-h-screen flex items-center justify-center bg-ink-950 relative overflow-hidden">

      {/* Ambient glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-10"
          style={{ background: 'radial-gradient(circle, #f59e0b 0%, transparent 70%)' }}
        />
      </div>

      <div className="relative z-10 w-full max-w-sm px-6">

        {/* Wordmark */}
        <div className="mb-10 text-center">
          <div className="inline-flex items-center gap-2 mb-3">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 shadow-[0_0_12px_#fbbf24]" />
            <span className="font-mono text-xs tracking-[0.3em] uppercase text-ink-400">ProcureBuddy</span>
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400 shadow-[0_0_12px_#fbbf24]" />
          </div>

          <h1 className="text-3xl font-light text-ink-50 tracking-tight">
            {step === 'login'     && 'Welcome back'}
            {step === 'reg-email' && 'Create account'}
            {step === 'reg-otp'  && 'Verify email'}
            {step === 'reg-done' && 'Account ready'}
          </h1>
          <p className="text-sm text-ink-400 mt-1">
            {step === 'login'     && 'Sign in to your workspace'}
            {step === 'reg-email' && 'Only *.cbri@csir.res.in emails allowed'}
            {step === 'reg-otp'  && `OTP sent to ${email}`}
            {step === 'reg-done' && 'You can now sign in'}
          </p>
        </div>

        <div className="bg-ink-900 border border-ink-700 rounded-2xl p-8 shadow-2xl">

          {/* ── LOGIN FORM ── */}
          {step === 'login' && (
            <form onSubmit={handleLogin} className="space-y-5">
              <div>
                <label className={labelCls}>Email</label>
                <input type="email" required value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="name.cbri@csir.res.in"
                  className={inputCls} />
              </div>
              <div>
                <label className={labelCls}>Password</label>
                <input type="password" required value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className={inputCls} />
              </div>

              {error && <p className="text-xs text-red-400 bg-red-950/40 border border-red-800/50 rounded-lg px-3 py-2">{error}</p>}

              <button type="submit" disabled={loading} className={btnCls}>
                {loading ? 'Signing in…' : 'Sign in'}
              </button>
            </form>
          )}

          {/* ── REGISTER STEP 1: email ── */}
          {step === 'reg-email' && (
            <form onSubmit={handleRegisterStart} className="space-y-5">
              <div>
                <label className={labelCls}>Official CBRI Email</label>
                <input type="email" required value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="name.cbri@csir.res.in"
                  className={inputCls} />
              </div>

              {error && <p className="text-xs text-red-400 bg-red-950/40 border border-red-800/50 rounded-lg px-3 py-2">{error}</p>}

              <button type="submit" disabled={loading} className={btnCls}>
                {loading ? 'Sending OTP…' : 'Send OTP'}
              </button>
            </form>
          )}

          {/* ── REGISTER STEP 2: OTP + password ── */}
          {step === 'reg-otp' && (
            <form onSubmit={handleRegisterVerify} className="space-y-5">
              <div>
                <label className={labelCls}>OTP Code</label>
                <input type="text" required maxLength={6} value={otp}
                  onChange={e => setOtp(e.target.value)}
                  placeholder="6-digit code"
                  className={inputCls + ' tracking-[0.5em] text-center font-mono text-lg'} />
                <p className="text-xs text-ink-500 mt-1.5">Check your email — valid for 10 minutes</p>
              </div>
              <div>
                <label className={labelCls}>Set Password</label>
                <input type="password" required value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Choose a strong password"
                  className={inputCls} />
              </div>

              {error && <p className="text-xs text-red-400 bg-red-950/40 border border-red-800/50 rounded-lg px-3 py-2">{error}</p>}

              <button type="submit" disabled={loading} className={btnCls}>
                {loading ? 'Verifying…' : 'Create Account'}
              </button>

              <button type="button"
                onClick={() => { setStep('reg-email'); setOtp(''); setError('') }}
                className="w-full text-xs text-ink-500 hover:text-amber-400 transition-colors pt-1">
                ← Resend OTP / change email
              </button>
            </form>
          )}

          {/* ── SUCCESS ── */}
          {step === 'reg-done' && (
            <div className="text-center space-y-5">
              <div className="w-14 h-14 mx-auto rounded-full bg-amber-500/15 border border-amber-500/30 flex items-center justify-center text-2xl">
                ✓
              </div>
              <p className="text-ink-300 text-sm">Account created successfully.<br />Sign in with your new credentials.</p>
              <button
                onClick={() => { setStep('login'); setOtp(''); setPassword(''); setError('') }}
                className={btnCls}>
                Go to Sign in
              </button>
            </div>
          )}

          {/* ── FOOTER TOGGLE ── */}
          {step !== 'reg-done' && (
            <div className="mt-6 text-center">
              {step === 'login' ? (
                <button
                  onClick={() => { setStep('reg-email'); setError('') }}
                  className="text-xs text-ink-400 hover:text-amber-400 transition-colors">
                  Don't have an account? Register
                </button>
              ) : (
                <button
                  onClick={() => { setStep('login'); setOtp(''); setPassword(''); setError('') }}
                  className="text-xs text-ink-400 hover:text-amber-400 transition-colors">
                  Already have an account? Sign in
                </button>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  )
}