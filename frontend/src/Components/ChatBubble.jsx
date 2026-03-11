export default function ChatBubble({ sender, text }) {
    const isUser = sender === "user";

    return (
        <div className={`flex mb-3 ${isUser ? "justify-end" : "justify-start"}`}>
            <div
                className={`max-w-[70%] p-3 rounded-xl ${
                    isUser
                        ? "bg-emerald-600 text-white"
                        : "bg-slate-100 text-slate-800 border border-slate-200"
                }`}
            >
                {text}
            </div>
        </div>
    );
}