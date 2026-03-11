import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createAccount } from "../api/api";

export default function Register() {
    const [form, setForm] = useState({ name: "", email: "", password: "" });
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setLoading(true);
        try {
            await createAccount(form);
            navigate("/totp");
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#0a0d12] flex items-center justify-center p-4">
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/3 right-1/4 w-80 h-80 bg-violet-500/7 rounded-full blur-3xl" />
                <div className="absolute bottom-1/3 left-1/4 w-64 h-64 bg-emerald-500/6 rounded-full blur-3xl" />
            </div>

            <div className="w-full max-w-sm relative">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-emerald-500 shadow-xl shadow-emerald-500/30 mb-4">
                        <svg className="w-6 h-6 text-black" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3z" />
                            <path d="M16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
                        </svg>
                    </div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">Create Account</h1>
                    <p className="text-gray-400 text-sm mt-1">Join CBRI ProcureBuddy today</p>
                </div>

                <div className="bg-white/8 backdrop-blur-xl border border-white/12 rounded-2xl p-6 shadow-2xl">
                    {error && (
                        <div className="mb-4 px-3 py-2.5 bg-red-500/15 border border-red-500/30 rounded-lg text-red-400 text-sm">
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1.5">Full name</label>
                            <input
                                type="text"
                                name="name"
                                value={form.name}
                                onChange={handleChange}
                                placeholder="Alex Rivera"
                                required
                                className="w-full px-3.5 py-2.5 bg-white/6 border border-white/12 rounded-xl text-white placeholder-gray-600 text-sm focus:outline-none focus:border-emerald-500/60 focus:bg-white/8 transition-all"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1.5">Email address</label>
                            <input
                                type="email"
                                name="email"
                                value={form.email}
                                onChange={handleChange}
                                placeholder="you@company.com"
                                required
                                className="w-full px-3.5 py-2.5 bg-white/6 border border-white/12 rounded-xl text-white placeholder-gray-600 text-sm focus:outline-none focus:border-emerald-500/60 focus:bg-white/8 transition-all"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1.5">Password</label>
                            <input
                                type="password"
                                name="password"
                                value={form.password}
                                onChange={handleChange}
                                placeholder="Min. 8 characters"
                                required
                                minLength={8}
                                className="w-full px-3.5 py-2.5 bg-white/6 border border-white/12 rounded-xl text-white placeholder-gray-600 text-sm focus:outline-none focus:border-emerald-500/60 focus:bg-white/8 transition-all"
                            />
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-2.5 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-60 disabled:cursor-not-allowed text-black font-semibold text-sm rounded-xl transition-all duration-200 shadow-lg shadow-emerald-500/20"
                        >
                            {loading ? "Creating account…" : "Create account"}
                        </button>
                    </form>

                    <div className="mt-4 pt-4 border-t border-white/8 text-center text-xs">
                        <span className="text-gray-500">Already have an account? </span>
                        <Link to="/" className="text-emerald-400 hover:text-emerald-300 transition-colors font-medium">
                            Sign in
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
}