import { useState } from "react";
import { Link } from "react-router-dom";

export default function ResetPassword() {
    const [email, setEmail] = useState("");
    const [sent, setSent] = useState(false);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        await new Promise((r) => setTimeout(r, 800));
        setSent(true);
        setLoading(false);
    };

    return (
        <div className="min-h-screen bg-[#0a0d12] flex items-center justify-center p-4">
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-80 h-80 bg-emerald-500/6 rounded-full blur-3xl" />
            </div>

            <div className="w-full max-w-sm relative">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-white/10 border border-white/15 mb-4">
                        <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                        </svg>
                    </div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">Reset Password</h1>
                    <p className="text-gray-400 text-sm mt-1">We'll send you a reset link</p>
                </div>

                <div className="bg-white/8 backdrop-blur-xl border border-white/12 rounded-2xl p-6 shadow-2xl">
                    {sent ? (
                        <div className="text-center py-4">
                            <div className="w-12 h-12 rounded-full bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center mx-auto mb-3">
                                <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                            </div>
                            <p className="text-white font-medium text-sm">Check your inbox</p>
                            <p className="text-gray-400 text-sm mt-1">
                                We sent a reset link to <span className="text-emerald-400">{email}</span>
                            </p>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1.5">Email address</label>
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@company.com"
                                    required
                                    className="w-full px-3.5 py-2.5 bg-white/6 border border-white/12 rounded-xl text-white placeholder-gray-600 text-sm focus:outline-none focus:border-emerald-500/60 focus:bg-white/8 transition-all"
                                />
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full py-2.5 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-60 text-black font-semibold text-sm rounded-xl transition-all duration-200 shadow-lg shadow-emerald-500/20"
                            >
                                {loading ? "Sending…" : "Send reset link"}
                            </button>
                        </form>
                    )}

                    <div className="mt-4 pt-4 border-t border-white/8 text-center text-xs">
                        <Link to="/" className="text-gray-400 hover:text-white transition-colors">
                            ← Back to sign in
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
}