import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { verifyTotp } from "../api/api";

export default function TotpVerify() {
    const [digits, setDigits] = useState(["", "", "", "", "", ""]);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const inputs = useRef([]);
    const navigate = useNavigate();

    const handleChange = (index, value) => {
        if (!/^\d*$/.test(value)) return;
        const newDigits = [...digits];
        newDigits[index] = value.slice(-1);
        setDigits(newDigits);
        if (value && index < 5) inputs.current[index + 1]?.focus();
    };

    const handleKeyDown = (index, e) => {
        if (e.key === "Backspace" && !digits[index] && index > 0) {
            inputs.current[index - 1]?.focus();
        }
    };

    const handlePaste = (e) => {
        const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
        if (pasted.length === 6) {
            setDigits(pasted.split(""));
            inputs.current[5]?.focus();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const code = digits.join("");
        if (code.length !== 6) return;
        setError("");
        setLoading(true);
        try {
            await verifyTotp(code);
            navigate("/app");
        } catch (err) {
            setError(err.message);
            setDigits(["", "", "", "", "", ""]);
            inputs.current[0]?.focus();
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#0a0d12] flex items-center justify-center p-4">
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-80 h-80 bg-violet-500/7 rounded-full blur-3xl" />
            </div>

            <div className="w-full max-w-sm relative">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-white/10 border border-white/15 mb-4">
                        <svg className="w-6 h-6 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                        </svg>
                    </div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">Two-Factor Auth</h1>
                    <p className="text-gray-400 text-sm mt-1">Enter the 6-digit code from your authenticator app</p>
                </div>

                <div className="bg-white/8 backdrop-blur-xl border border-white/12 rounded-2xl p-6 shadow-2xl">
                    {error && (
                        <div className="mb-4 px-3 py-2.5 bg-red-500/15 border border-red-500/30 rounded-lg text-red-400 text-sm text-center">
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit}>
                        <div className="flex gap-2 justify-center mb-6" onPaste={handlePaste}>
                            {digits.map((digit, i) => (
                                <input
                                    key={i}
                                    ref={(el) => (inputs.current[i] = el)}
                                    type="text"
                                    inputMode="numeric"
                                    maxLength={1}
                                    value={digit}
                                    onChange={(e) => handleChange(i, e.target.value)}
                                    onKeyDown={(e) => handleKeyDown(i, e)}
                                    className="w-11 h-13 text-center text-xl font-bold bg-white/6 border border-white/12 rounded-xl text-white focus:outline-none focus:border-violet-500/70 focus:bg-white/10 transition-all caret-transparent"
                                    style={{ height: "52px" }}
                                />
                            ))}
                        </div>

                        <button
                            type="submit"
                            disabled={loading || digits.join("").length !== 6}
                            className="w-full py-2.5 bg-violet-500 hover:bg-violet-400 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-sm rounded-xl transition-all duration-200 shadow-lg shadow-violet-500/20"
                        >
                            {loading ? "Verifying…" : "Verify code"}
                        </button>
                    </form>

                    <p className="text-center text-xs text-gray-500 mt-4">
                        Didn't receive it?{" "}
                        <button className="text-violet-400 hover:text-violet-300 transition-colors">Resend code</button>
                    </p>
                </div>
            </div>
        </div>
    );
}