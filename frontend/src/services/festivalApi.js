/**
 * Festival API Service — Calendarific Integration
 *
 * Fetches Indian festivals from Calendarific API, caches per session,
 * and provides detection logic for ±3 day theme activation.
 *
 * API call: once per session (or once per day). Cached in sessionStorage.
 */

const API_KEY = '103AdTHyHdtcJwER9A3SvF0IijuwDIht';
const BASE_URL = 'https://calendarific.com/api/v2/holidays';
const CACHE_KEY = 'procurebuddy-festival-cache';
const POPUP_SHOWN_KEY = 'procurebuddy-festival-popup-shown';

// ── Priority system ─────────────────────────────────────────────────────
// Higher = more important. If two festivals overlap, highest priority wins.
const PRIORITY = {
  'diwali':              5,
  'holi':                4,
  'ramzan id':           4,
  'eid':                 4,
  'bakrid':              4,
  'independence day':    3,
  'republic day':        3,
  'navratri':            3,
  'dussehra':            3,
  'ganesh chaturthi':    3,
  'janmashtami':         2,
  'christmas':           2,
  'guru nanak jayanti':  2,
  'raksha bandhan':      2,
  'buddha purnima':      2,
  'mahavir jayanti':     2,
  'muharram':            2,
  'milad un-nabi':       2,
  'onam':                2,
  'pongal':              1,
  'makar sankranti':     1,
  'good friday':         1,
  'chhath':              1,
  'bhai duj':            1,
};

// ── Theme mapping ───────────────────────────────────────────────────────
// Maps festival name (lowercase match) → theme class + CSS palette.
const THEMES = {
  'diwali':            { class: 'dark-gold',  accent: '#f97316', glowA: 'rgba(249,115,22,0.24)',  glowB: 'rgba(250,204,21,0.2)',   sparkle: '#fde68a' },
  'holi':              { class: 'colorful',   accent: '#ec4899', glowA: 'rgba(236,72,153,0.22)',  glowB: 'rgba(34,197,94,0.22)',   sparkle: '#f9a8d4' },
  'ramzan id':         { class: 'green',      accent: '#0891b2', glowA: 'rgba(8,145,178,0.24)',   glowB: 'rgba(22,163,74,0.18)',   sparkle: '#bae6fd' },
  'bakrid':            { class: 'green',      accent: '#0891b2', glowA: 'rgba(8,145,178,0.24)',   glowB: 'rgba(22,163,74,0.18)',   sparkle: '#bae6fd' },
  'independence day':  { class: 'tricolor',   accent: '#f97316', glowA: 'rgba(249,115,22,0.22)',  glowB: 'rgba(34,197,94,0.22)',   sparkle: '#fdba74' },
  'republic day':      { class: 'tricolor',   accent: '#f97316', glowA: 'rgba(249,115,22,0.22)',  glowB: 'rgba(34,197,94,0.22)',   sparkle: '#fdba74' },
  'navratri':          { class: 'festive-red', accent: '#f15b6c', glowA: 'rgba(244,114,182,0.22)', glowB: 'rgba(249,115,22,0.2)',  sparkle: '#ffd66b' },
  'dussehra':          { class: 'festive-red', accent: '#f15b6c', glowA: 'rgba(244,114,182,0.22)', glowB: 'rgba(249,115,22,0.2)',  sparkle: '#ffd66b' },
  'ganesh chaturthi':  { class: 'orange',     accent: '#ea580c', glowA: 'rgba(234,88,12,0.24)',   glowB: 'rgba(249,115,22,0.22)', sparkle: '#fdba74' },
  'janmashtami':       { class: 'blue',       accent: '#2563eb', glowA: 'rgba(37,99,235,0.22)',   glowB: 'rgba(14,165,233,0.18)', sparkle: '#bfdbfe' },
  'christmas':         { class: 'green-red',  accent: '#16a34a', glowA: 'rgba(22,163,74,0.2)',    glowB: 'rgba(220,38,38,0.18)',  sparkle: '#dcfce7' },
  'guru nanak jayanti':{ class: 'purple',     accent: '#7c3aed', glowA: 'rgba(124,58,237,0.22)',  glowB: 'rgba(59,130,246,0.18)', sparkle: '#ddd6fe' },
  'raksha bandhan':    { class: 'warm',       accent: '#ef4444', glowA: 'rgba(239,68,68,0.22)',   glowB: 'rgba(245,158,11,0.18)', sparkle: '#fde68a' },
  'buddha purnima':    { class: 'blue',       accent: '#0284c7', glowA: 'rgba(2,132,199,0.22)',   glowB: 'rgba(56,189,248,0.18)', sparkle: '#bae6fd' },
  'mahavir jayanti':   { class: 'gold',       accent: '#a16207', glowA: 'rgba(161,98,7,0.22)',    glowB: 'rgba(251,191,36,0.18)', sparkle: '#fde68a' },
  'muharram':          { class: 'teal',       accent: '#0f766e', glowA: 'rgba(15,118,110,0.22)',  glowB: 'rgba(6,182,212,0.18)',  sparkle: '#99f6e4' },
  'milad un-nabi':     { class: 'sky',        accent: '#0ea5e9', glowA: 'rgba(14,165,233,0.22)',  glowB: 'rgba(34,197,94,0.18)',  sparkle: '#bae6fd' },
  'onam':              { class: 'gold',       accent: '#ca8a04', glowA: 'rgba(202,138,4,0.24)',   glowB: 'rgba(245,158,11,0.2)',  sparkle: '#fde68a' },
  'pongal':            { class: 'warm',       accent: '#f59e0b', glowA: 'rgba(56,189,248,0.22)',  glowB: 'rgba(245,158,11,0.22)',sparkle: '#fde68a' },
  'makar sankranti':   { class: 'warm',       accent: '#f59e0b', glowA: 'rgba(56,189,248,0.22)',  glowB: 'rgba(245,158,11,0.22)',sparkle: '#fde68a' },
  'good friday':       { class: 'indigo',     accent: '#6366f1', glowA: 'rgba(99,102,241,0.22)',  glowB: 'rgba(148,163,184,0.18)',sparkle: '#c7d2fe' },
  'chhath':            { class: 'amber',      accent: '#f59e0b', glowA: 'rgba(245,158,11,0.24)',  glowB: 'rgba(249,115,22,0.2)', sparkle: '#fde68a' },
};

const DEFAULT_THEME = { class: 'light', accent: '#355cff', glowA: 'rgba(53,92,255,0.12)', glowB: 'rgba(53,92,255,0.08)', sparkle: '#bfdbfe' };

// ── Helpers ─────────────────────────────────────────────────────────────

/** Get today's date string in YYYY-MM-DD (IST). */
function getTodayIST() {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(new Date());
}

/** Compute day difference between two YYYY-MM-DD strings. */
function dayDiff(dateA, dateB) {
  const a = new Date(dateA + 'T00:00:00Z');
  const b = new Date(dateB + 'T00:00:00Z');
  return Math.round((b - a) / 86400000);
}

/** Match a festival name to our priority/theme maps (fuzzy lowercase). */
function matchFestivalKey(name) {
  const lower = name.toLowerCase();
  // Direct match
  for (const key of Object.keys(PRIORITY)) {
    if (lower.includes(key)) return key;
  }
  // Partial matches for API name variations
  if (lower.includes('diwali') || lower.includes('deepavali')) return 'diwali';
  if (lower.includes('holi'))      return 'holi';
  if (lower.includes('eid') || lower.includes('ramzan'))  return 'ramzan id';
  if (lower.includes('navratri'))  return 'navratri';
  if (lower.includes('chhat') || lower.includes('chhath'))return 'chhath';
  if (lower.includes('guru nanak'))return 'guru nanak jayanti';
  return null;
}

// ── API fetch with session cache ────────────────────────────────────────

/**
 * Fetch festivals from Calendarific API. Cached in sessionStorage for
 * the current day — no repeated API calls.
 */
export async function fetchFestivals(year) {
  const cacheRaw = sessionStorage.getItem(CACHE_KEY);
  if (cacheRaw) {
    try {
      const cached = JSON.parse(cacheRaw);
      if (cached.year === year && cached.fetchDate === getTodayIST()) {
        console.info('[FestivalAPI] Using cached festival data');
        return cached.festivals;
      }
    } catch { /* cache corrupt, refetch */ }
  }

  try {
    const url = `${BASE_URL}?api_key=${API_KEY}&country=IN&year=${year}&type=religious,national`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`API ${res.status}`);

    const data = await res.json();
    const holidays = data?.response?.holidays ?? [];

    const festivals = holidays.map((h) => ({
      name: h.name,
      date: h.date?.iso,
      description: h.description || '',
      type: h.primary_type || '',
      key: matchFestivalKey(h.name),
    })).filter((f) => f.key !== null);

    // Deduplicate by key+date (API sometimes returns same festival twice)
    const seen = new Set();
    const unique = festivals.filter((f) => {
      const id = `${f.key}:${f.date}`;
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    });

    sessionStorage.setItem(CACHE_KEY, JSON.stringify({
      year,
      fetchDate: getTodayIST(),
      festivals: unique,
    }));

    console.info(`[FestivalAPI] Fetched ${unique.length} festivals for ${year}`);
    return unique;
  } catch (err) {
    console.warn('[FestivalAPI] Fetch failed, using fallback:', err.message);
    return []; // Fallback: empty → default theme, no crash
  }
}

// ── Festival detection (±3 days) ────────────────────────────────────────

/**
 * Detect the active festival for today.
 *
 * Returns: { festival, theme, isExactDay, daysUntil } or null.
 * - Checks ±3 days window for pre-theme activation.
 * - Picks highest priority if multiple festivals are within range.
 * - No rapid switching: single winner only.
 */
export function detectActiveFestival(festivals) {
  const today = getTodayIST();
  const LEAD_DAYS = 3;

  // Find all festivals within ±3 days
  const candidates = [];
  for (const f of festivals) {
    const diff = dayDiff(today, f.date);
    if (diff >= -1 && diff <= LEAD_DAYS) {
      // diff=0 → exact day, diff>0 → upcoming (pre-theme), diff<0 → just passed
      candidates.push({
        ...f,
        daysUntil: diff,
        isExactDay: diff === 0,
        priority: PRIORITY[f.key] ?? 0,
      });
    }
  }

  if (candidates.length === 0) return null;

  // Sort: highest priority first, then closest date
  candidates.sort((a, b) => b.priority - a.priority || a.daysUntil - b.daysUntil);

  const winner = candidates[0];
  const theme = THEMES[winner.key] || DEFAULT_THEME;

  return {
    festival: winner,
    theme,
    isExactDay: winner.isExactDay,
    daysUntil: winner.daysUntil,
  };
}

// ── Theme application ───────────────────────────────────────────────────

/**
 * Apply festival theme to the document root. Smooth transition via
 * CSS transition property on root.
 */
export function applyFestivalTheme(result) {
  const root = document.documentElement;

  // Enable smooth transitions
  root.style.transition = 'background-color 0.6s ease, color 0.3s ease';

  if (!result) {
    // No active festival → default theme
    root.dataset.festivalTheme = 'none';
    root.style.removeProperty('--festival-accent');
    root.style.removeProperty('--festival-glow-a');
    root.style.removeProperty('--festival-glow-b');
    root.style.removeProperty('--festival-sparkle');
    root.classList.remove('festival-active', 'festival-pre');
    return;
  }

  const { theme, isExactDay, festival } = result;

  root.dataset.festivalTheme = theme.class;
  root.dataset.festivalName = festival.name;
  root.style.setProperty('--festival-accent', theme.accent);
  root.style.setProperty('--festival-glow-a', theme.glowA);
  root.style.setProperty('--festival-glow-b', theme.glowB);
  root.style.setProperty('--festival-sparkle', theme.sparkle);

  if (isExactDay) {
    root.classList.add('festival-active');
    root.classList.remove('festival-pre');
  } else {
    root.classList.add('festival-pre');
    root.classList.remove('festival-active');
  }

  console.info(`[FestivalAPI] Applied theme: ${theme.class} for ${festival.name}` +
    (isExactDay ? ' (TODAY!)' : ` (in ${result.daysUntil} days)`));
}

// ── Popup logic ─────────────────────────────────────────────────────────

/**
 * Check if festival popup should be shown. Returns festival name or null.
 * Shows only once per session per festival.
 */
export function shouldShowPopup(result) {
  if (!result || !result.isExactDay) return null;

  const shownRaw = sessionStorage.getItem(POPUP_SHOWN_KEY);
  const shown = shownRaw ? JSON.parse(shownRaw) : [];

  const popupId = `${result.festival.key}:${result.festival.date}`;
  if (shown.includes(popupId)) return null;

  // Mark as shown
  shown.push(popupId);
  sessionStorage.setItem(POPUP_SHOWN_KEY, JSON.stringify(shown));

  return result.festival.name;
}
