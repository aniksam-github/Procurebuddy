import { createContext, useContext, useEffect, useMemo, useState } from 'react';

const THEME_KEY = 'procurebuddy-theme';
const UI_MODE_KEY = 'procurebuddy-ui-mode';
const FONT_KEY = 'procurebuddy-font';
const ACCENT_KEY = 'procurebuddy-accent';

const FONT_MAP = {
  inter: "'Inter', system-ui, sans-serif",
  satoshi: "'Satoshi', 'Inter', system-ui, sans-serif",
  system: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
};

const ACCENT_MAP = {
  blue: {
    label: 'Electric Blue',
    light: {
      accent: '#355cff',
      accentStrong: '#2346e6',
      brandA: '#74d8e8',
      brandB: '#355cff',
      brandC: '#8d7dff',
      accentRgb: '53 92 255',
    },
    dark: {
      accent: '#6f8fff',
      accentStrong: '#8aa3ff',
      brandA: '#7dd3fc',
      brandB: '#6f8fff',
      brandC: '#7f7cff',
      accentRgb: '111 143 255',
    },
  },
  emerald: {
    label: 'Emerald',
    light: {
      accent: '#169c67',
      accentStrong: '#0d8554',
      brandA: '#79e2bf',
      brandB: '#169c67',
      brandC: '#4dc7a1',
      accentRgb: '22 156 103',
    },
    dark: {
      accent: '#43d79b',
      accentStrong: '#67e2af',
      brandA: '#91f0cf',
      brandB: '#43d79b',
      brandC: '#2db984',
      accentRgb: '67 215 155',
    },
  },
  teal: {
    label: 'Teal',
    light: {
      accent: '#0f9f9a',
      accentStrong: '#0b8682',
      brandA: '#76e4de',
      brandB: '#0f9f9a',
      brandC: '#4bc3f7',
      accentRgb: '15 159 154',
    },
    dark: {
      accent: '#44d4ce',
      accentStrong: '#68e6e0',
      brandA: '#8ff1eb',
      brandB: '#44d4ce',
      brandC: '#67b8ff',
      accentRgb: '68 212 206',
    },
  },
  amber: {
    label: 'Amber',
    light: {
      accent: '#d97706',
      accentStrong: '#b85f00',
      brandA: '#ffd27d',
      brandB: '#d97706',
      brandC: '#ff9f43',
      accentRgb: '217 119 6',
    },
    dark: {
      accent: '#f3a537',
      accentStrong: '#ffbf67',
      brandA: '#ffd67e',
      brandB: '#f3a537',
      brandC: '#ff9f43',
      accentRgb: '243 165 55',
    },
  },
  coral: {
    label: 'Coral',
    light: {
      accent: '#ef6a4b',
      accentStrong: '#d45538',
      brandA: '#ffb08d',
      brandB: '#ef6a4b',
      brandC: '#ff8a72',
      accentRgb: '239 106 75',
    },
    dark: {
      accent: '#ff8f73',
      accentStrong: '#ffab94',
      brandA: '#ffc1ad',
      brandB: '#ff8f73',
      brandC: '#ffb36a',
      accentRgb: '255 143 115',
    },
  },
  violet: {
    label: 'Violet',
    light: {
      accent: '#7c5cff',
      accentStrong: '#6545e8',
      brandA: '#b7a8ff',
      brandB: '#7c5cff',
      brandC: '#78b7ff',
      accentRgb: '124 92 255',
    },
    dark: {
      accent: '#a58cff',
      accentStrong: '#b8a5ff',
      brandA: '#d0c2ff',
      brandB: '#a58cff',
      brandC: '#82c0ff',
      accentRgb: '165 140 255',
    },
  },
  rose: {
    label: 'Rose',
    light: {
      accent: '#df4d7a',
      accentStrong: '#c43763',
      brandA: '#ff9eb8',
      brandB: '#df4d7a',
      brandC: '#ff7a59',
      accentRgb: '223 77 122',
    },
    dark: {
      accent: '#ff7ca1',
      accentStrong: '#ff9ab6',
      brandA: '#ffc0cf',
      brandB: '#ff7ca1',
      brandC: '#ff9b7a',
      accentRgb: '255 124 161',
    },
  },
  slate: {
    label: 'Slate',
    light: {
      accent: '#51607a',
      accentStrong: '#3f4c63',
      brandA: '#94a3b8',
      brandB: '#51607a',
      brandC: '#6d86a8',
      accentRgb: '81 96 122',
    },
    dark: {
      accent: '#94a3b8',
      accentStrong: '#b0bccd',
      brandA: '#d3dbe6',
      brandB: '#94a3b8',
      brandC: '#7e93b3',
      accentRgb: '148 163 184',
    },
  },
};

const ThemeContext = createContext(null);

function getStoredValue(key, fallback) {
  if (typeof window === 'undefined') {
    return fallback;
  }
  return window.localStorage.getItem(key) || fallback;
}

function resolveTheme(theme) {
  if (typeof window === 'undefined') {
    return 'light';
  }
  if (theme !== 'system') {
    return theme;
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => getStoredValue(THEME_KEY, 'light'));
  const [uiMode, setUiMode] = useState(() => getStoredValue(UI_MODE_KEY, 'minimal'));
  const [fontFamily, setFontFamily] = useState(() => getStoredValue(FONT_KEY, 'inter'));
  const [accentColor, setAccentColor] = useState(() => getStoredValue(ACCENT_KEY, 'blue'));
  const [systemTheme, setSystemTheme] = useState(() => resolveTheme('system'));

  const resolvedTheme = theme === 'system' ? systemTheme : theme;

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const sync = () => setSystemTheme(media.matches ? 'dark' : 'light');

    sync();
    media.addEventListener('change', sync);
    return () => media.removeEventListener('change', sync);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(UI_MODE_KEY, uiMode);
    document.documentElement.dataset.uiMode = uiMode;
  }, [uiMode]);

  useEffect(() => {
    window.localStorage.setItem(FONT_KEY, fontFamily);
    document.documentElement.style.setProperty('--font-body', FONT_MAP[fontFamily] || FONT_MAP.inter);
    document.documentElement.dataset.font = fontFamily;
  }, [fontFamily]);

  useEffect(() => {
    window.localStorage.setItem(ACCENT_KEY, accentColor);
  }, [accentColor]);

  useEffect(() => {
    const palette = ACCENT_MAP[accentColor] || ACCENT_MAP.blue;
    const tones = palette[resolvedTheme] || palette.light;
    const softAlpha = uiMode === 'futuristic' ? (resolvedTheme === 'dark' ? 0.22 : 0.16) : (resolvedTheme === 'dark' ? 0.18 : 0.12);
    const lightAlpha = uiMode === 'futuristic' ? (resolvedTheme === 'dark' ? 0.3 : 0.24) : (resolvedTheme === 'dark' ? 0.2 : 0.18);
    const ringAlpha = resolvedTheme === 'dark' ? 0.26 : 0.24;

    document.documentElement.style.setProperty('--accent', tones.accent);
    document.documentElement.style.setProperty('--accent-strong', tones.accentStrong);
    document.documentElement.style.setProperty('--accent-soft', `rgb(${tones.accentRgb} / ${softAlpha})`);
    document.documentElement.style.setProperty('--accent-light', `rgb(${tones.accentRgb} / ${lightAlpha})`);
    document.documentElement.style.setProperty('--brand-a', tones.brandA);
    document.documentElement.style.setProperty('--brand-b', tones.brandB);
    document.documentElement.style.setProperty('--brand-c', tones.brandC);
    document.documentElement.style.setProperty('--ring', `rgb(${tones.accentRgb} / ${ringAlpha})`);
    document.documentElement.style.setProperty('--accent-rgb', tones.accentRgb);
    document.documentElement.dataset.accent = accentColor;
  }, [accentColor, resolvedTheme, uiMode]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', resolvedTheme === 'dark');
    document.documentElement.dataset.theme = resolvedTheme;
  }, [resolvedTheme]);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      resolvedTheme,
      uiMode,
      setUiMode,
      fontFamily,
      setFontFamily,
      accentColor,
      setAccentColor,
      accentOptions: ACCENT_MAP,
    }),
    [theme, resolvedTheme, uiMode, fontFamily, accentColor]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}
