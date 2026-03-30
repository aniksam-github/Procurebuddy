import React, { Suspense, lazy } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import SeasonalLayer from './SeasonalLayer';
import { SeasonalProvider } from './context/SeasonalContext';
import { ThemeProvider } from './context/ThemeContext';
import './index.css';

const Home = lazy(() => import('./Home.jsx'));
const App = lazy(() => import('./App.jsx'));

function AppLoader() {
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="rounded-[24px] border border-[color:var(--border-soft)] bg-[color:var(--card-strong)] px-5 py-4 text-sm text-[color:var(--text-secondary)] shadow-soft">
        Loading workspace...
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider>
      <SeasonalProvider>
        <SeasonalLayer />
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Suspense fallback={<AppLoader />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/chat" element={<App />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </SeasonalProvider>
    </ThemeProvider>
  </React.StrictMode>
);
