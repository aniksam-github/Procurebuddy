import { createContext, useContext, useState } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [email, setEmail] = useState(() => localStorage.getItem('userEmail') || null)

  function login(userEmail) {
    localStorage.setItem('userEmail', userEmail)
    setEmail(userEmail)
  }

  function logout() {
    localStorage.removeItem('userEmail')
    setEmail(null)
  }

  return (
    <AuthContext.Provider value={{ email, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}