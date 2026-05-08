/**
 * useFestivalApi — React hook for Calendarific festival integration.
 *
 * Fetches festivals once on mount, detects active festival, applies theme,
 * and triggers popup. No polling, no rapid switching.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  applyFestivalTheme,
  detectActiveFestival,
  fetchFestivals,
  shouldShowPopup,
} from '../services/festivalApi';

/**
 * @returns {{
 *   activeFestival: object|null,
 *   festivalTheme: object|null,
 *   isExactDay: boolean,
 *   popupMessage: string|null,
 *   dismissPopup: () => void,
 *   loading: boolean,
 *   error: string|null,
 * }}
 */
export function useFestivalApi() {
  const [activeFestival, setActiveFestival] = useState(null);
  const [festivalTheme, setFestivalTheme] = useState(null);
  const [isExactDay, setIsExactDay] = useState(false);
  const [popupMessage, setPopupMessage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const initialized = useRef(false);

  useEffect(() => {
    // Run once per mount — no repeated calls
    if (initialized.current) return;
    initialized.current = true;

    async function init() {
      try {
        setLoading(true);
        const year = new Date().getFullYear();
        const festivals = await fetchFestivals(year);

        const result = detectActiveFestival(festivals);
        applyFestivalTheme(result);

        if (result) {
          setActiveFestival(result.festival);
          setFestivalTheme(result.theme);
          setIsExactDay(result.isExactDay);

          const name = shouldShowPopup(result);
          if (name) {
            setPopupMessage(`🎉 Happy ${name}!`);
          }
        }
      } catch (err) {
        console.warn('[useFestivalApi] Init failed:', err.message);
        setError(err.message);
        applyFestivalTheme(null); // fallback to default
      } finally {
        setLoading(false);
      }
    }

    init();
  }, []);

  const dismissPopup = useCallback(() => setPopupMessage(null), []);

  return {
    activeFestival,
    festivalTheme,
    isExactDay,
    popupMessage,
    dismissPopup,
    loading,
    error,
  };
}
