import { createContext, useContext, useEffect, useMemo, useState } from 'react';

const SEASONAL_MODE_KEY = 'procurebuddy-seasonal-mode';

const NAVRATRI_THEME = {
  id: 'navratri',
  name: 'Navratri',
  accent: '#f15b6c',
  palette: {
    glowA: 'rgba(244, 114, 182, 0.22)',
    glowB: 'rgba(249, 115, 22, 0.2)',
    glowC: 'rgba(251, 191, 36, 0.16)',
    sparkle: '#ffd66b',
    ornament: 'rgba(241, 91, 108, 0.18)',
    border: 'rgba(241, 91, 108, 0.2)',
  },
};

const SeasonalContext = createContext(null);

function getStoredMode() {
  if (typeof window === 'undefined') {
    return 'auto';
  }

  const value = window.localStorage.getItem(SEASONAL_MODE_KEY);
  if (value === 'on') {
    return 'always';
  }
  return value === 'always' || value === 'off' || value === 'auto' ? value : 'auto';
}

function resolveSeasonalState(mode) {
  if (mode === 'off') {
    return {
      mode,
      isEnabled: false,
      activeFestival: null,
      intensity: 'off',
      debugLabel: 'Seasonal layer disabled',
    };
  }

  if (mode === 'always') {
    return {
      mode: 'always',
      isEnabled: true,
      activeFestival: NAVRATRI_THEME,
      intensity: 'strong',
      debugLabel: 'Seasonal layer forced on',
    };
  }

  return {
    mode: 'auto',
    isEnabled: true,
    activeFestival: NAVRATRI_THEME,
    intensity: 'soft',
    debugLabel: `Auto-detected festival: ${NAVRATRI_THEME.name}`,
  };
}

export function SeasonalProvider({ children }) {
  const [mode, setModeState] = useState(getStoredMode);
  const resolved = useMemo(() => resolveSeasonalState(mode), [mode]);

  useEffect(() => {
    window.localStorage.setItem(SEASONAL_MODE_KEY, mode);
    document.documentElement.dataset.seasonalMode = mode;
    document.documentElement.dataset.seasonalFestival = resolved.activeFestival?.id || 'none';
    console.info('[SeasonalProvider] mode changed', {
      mode,
      enabled: resolved.isEnabled,
      festival: resolved.activeFestival?.name ?? 'none',
      intensity: resolved.intensity,
    });
  }, [mode, resolved]);

  function setMode(nextMode) {
    const normalizedMode = nextMode === 'on' ? 'always' : nextMode;
    console.info('[SeasonalProvider] setMode called', { nextMode: normalizedMode });
    setModeState(normalizedMode);
  }

  const value = useMemo(
    () => ({
      mode,
      setMode,
      isEnabled: resolved.isEnabled,
      activeFestival: resolved.activeFestival,
      intensity: resolved.intensity,
      debugLabel: resolved.debugLabel,
    }),
    [mode, resolved]
  );

  return <SeasonalContext.Provider value={value}>{children}</SeasonalContext.Provider>;
}

export function useSeasonal() {
  const context = useContext(SeasonalContext);
  if (!context) {
    throw new Error('useSeasonal must be used within SeasonalProvider');
  }
  return context;
}
