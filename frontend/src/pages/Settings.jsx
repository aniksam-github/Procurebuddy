import { enable2FA, disable2FA } from "../api/api";

export default function Settings() {
    return (
        <div className="max-w-xl space-y-6">
            <h2 className="text-2xl font-bold text-slate-800">Settings</h2>

            <div className="bg-white p-4 rounded-xl border border-slate-200">
                <h3 className="font-semibold mb-2 text-slate-800">Profile</h3>
                <p className="text-slate-600">Name: Aniket Samanta</p>
                <p className="text-slate-600">Email: 245401006@gkv.ac.in</p>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200">
                <h3 className="font-semibold mb-3 text-slate-800">Security</h3>
                <div className="flex gap-3">
                    <button
                        onClick={enable2FA}
                        className="bg-emerald-600 hover:bg-emerald-700 transition px-4 py-2 rounded text-white"
                    >
                        Enable 2FA
                    </button>
                    <button
                        onClick={disable2FA}
                        className="bg-red-500 hover:bg-red-600 transition px-4 py-2 rounded text-white"
                    >
                        Disable 2FA
                    </button>
                </div>
            </div>
        </div>
    );
}