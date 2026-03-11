import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchChats } from "../api/api";

export default function Sidebar() {
    const [chats, setChats] = useState([]);

    useEffect(() => {
        fetchChats().then(setChats);
    }, []);

    return (
        <div className="w-64 bg-white border-r border-slate-200 p-4 flex flex-col">
            <h1 className="text-xl font-bold mb-4 text-slate-800">ProcureBuddy</h1>

            <button className="w-full mb-4 bg-emerald-600 hover:bg-emerald-700 transition rounded-lg p-2 text-white font-semibold">
                + New Chat
            </button>

            <div className="flex-1 overflow-y-auto space-y-2 text-sm">
                {chats.map((chat) => (
                    <div
                        key={chat.id}
                        className="p-2 rounded-lg cursor-pointer bg-slate-100 hover:bg-slate-200 text-slate-700"
                    >
                        {chat.title}
                    </div>
                ))}
            </div>

            <Link
                to="/settings"
                className="mt-4 text-sm text-slate-600 hover:text-slate-900"
            >
                ⚙️ Settings
            </Link>
        </div>
    );
}