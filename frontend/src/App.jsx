import { Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login.jsx'
import Chat from './pages/Chat.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/chat" element={<Chat />} />
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  )
}