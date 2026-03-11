import { Outlet } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Topbar from "../components/Topbar";

export default function AppLayout() {
    return (
        <div className="flex h-screen bg-slate-100 text-slate-900">
            <Sidebar />
            <div className="flex flex-col flex-1">
                <Topbar />
                <div className="flex-1 p-6 overflow-hidden">
                    <Outlet />
                </div>
            </div>
        </div>
    );
}