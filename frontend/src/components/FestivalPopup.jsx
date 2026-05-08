/**
 * FestivalPopup — Shows a celebratory popup on the exact festival day.
 * Displays once per session, auto-dismisses after 6 seconds,
 * or user can click to close.
 */

import { AnimatePresence, motion } from 'framer-motion';
import { useEffect } from 'react';

export function FestivalPopup({ message, onDismiss }) {
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(onDismiss, 6000);
    return () => clearTimeout(timer);
  }, [message, onDismiss]);

  return (
    <AnimatePresence>
      {message && (
        <motion.div
          initial={{ opacity: 0, y: -40, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -30, scale: 0.95 }}
          transition={{ type: 'spring', stiffness: 300, damping: 24 }}
          onClick={onDismiss}
          style={{
            position: 'fixed',
            top: '24px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 9999,
            padding: '16px 32px',
            borderRadius: '16px',
            background: 'var(--festival-accent, #f97316)',
            color: '#fff',
            fontSize: '1.1rem',
            fontWeight: 600,
            letterSpacing: '0.02em',
            boxShadow: '0 8px 32px rgba(0,0,0,0.25), 0 0 60px var(--festival-glow-a, rgba(249,115,22,0.3))',
            cursor: 'pointer',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.2)',
            textAlign: 'center',
            maxWidth: '90vw',
            userSelect: 'none',
          }}
        >
          {message}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
