export default function Topbar() {
    return (
        <div className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-4">
            <div className="font-semibold text-slate-800">CBRI ProcureBuddy</div>
            <button className="text-sm bg-red-500 hover:bg-red-600 px-3 py-1 rounded text-white">
                Logout
            </button>
        </div>
    );
}