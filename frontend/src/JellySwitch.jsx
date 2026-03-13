import { useRef, useEffect } from 'react';
import { SwitchBehavior } from './physics';

// ─────────────────────────────────────────────────
//  JellySwitch.jsx
//  Canvas-rendered toggle with TypeGPU spring physics
//  Props:
//    checked  : boolean
//    onChange : (bool) => void
//    color    : string  (hex, default accent)
//    size     : number  (scale multiplier, default 1)
// ─────────────────────────────────────────────────

function getThemeValue(name, fallback) {
  if (typeof window === 'undefined') {
    return fallback;
  }
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export default function JellySwitch({ checked, onChange, color, size = 1 }) {
  const cRef    = useRef(null);
  const swRef   = useRef(null);
  const rafRef  = useRef(null);
  const tsRef   = useRef(null);
  const colorRef = useRef(color || getThemeValue('--accent', '#b24b7d'));
  const railRef = useRef(getThemeValue('--input-bg', '#d4d8eb'));

  const W = Math.round(72 * size);
  const H = Math.round(38 * size);

  if (!swRef.current) swRef.current = new SwitchBehavior();

  useEffect(() => { swRef.current.toggled = checked; }, [checked]);
  useEffect(() => {
    colorRef.current = color || getThemeValue('--accent', '#b24b7d');
    railRef.current = getThemeValue('--input-bg', '#d4d8eb');
  }, [color]);

  useEffect(() => {
    const canvas = cRef.current;
    const ctx = canvas.getContext('2d');

    function rr(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r, y); ctx.lineTo(x+w-r, y); ctx.arcTo(x+w, y, x+w, y+r, r);
      ctx.lineTo(x+w, y+h-r); ctx.arcTo(x+w, y+h, x+w-r, y+h, r);
      ctx.lineTo(x+r, y+h); ctx.arcTo(x, y+h, x, y+h-r, r);
      ctx.lineTo(x, y+r); ctx.arcTo(x, y, x+r, y, r);
      ctx.closePath();
    }

    function frame(ts) {
      if (!tsRef.current) tsRef.current = ts;
      const dt = Math.min((ts - tsRef.current) * 0.001, 0.05);
      tsRef.current = ts;
      swRef.current.update(dt);
      const s = swRef.current.state;
      const c = colorRef.current;

      const pad = 3.5 * size, rW = W - pad*2, rH = H * 0.52;
      const rx = pad, ry = (H - rH) / 2, cornerR = rH / 2;
      ctx.clearRect(0, 0, W, H);

      // Rail background
      rr(rx, ry, rW, rH, cornerR);
      ctx.fillStyle = railRef.current;
      ctx.fill();

      // Colored fill
      if (s.progress > 0.005) {
        const fw = cornerR + s.progress * (rW - cornerR * 2);
        rr(rx, ry, fw + cornerR, rH, cornerR);
        const g = ctx.createLinearGradient(rx, 0, rx + rW, 0);
        g.addColorStop(0, c + '99'); g.addColorStop(1, c);
        ctx.fillStyle = g; ctx.fill();
      }

      // Thumb with spring deformation
      const thumbR = rH * 0.46;
      const thumbX = rx + cornerR + s.progress * (rW - cornerR * 2);
      const thumbY = H / 2;
      const scaleX = Math.max(0.45, 1 - s.squashX * 0.30);
      const scaleY = Math.max(0.45, 1 + s.squashZ * 0.16);
      const angle  = s.wiggleX * 0.065;

      ctx.save();
      ctx.translate(thumbX, thumbY);
      ctx.rotate(angle);
      ctx.scale(scaleX, scaleY);
      ctx.shadowColor = c + '55'; ctx.shadowBlur = 10 * size; ctx.shadowOffsetY = 2 * size;
      ctx.beginPath(); ctx.arc(0, 0, thumbR, 0, Math.PI * 2);
      ctx.fillStyle = 'white'; ctx.fill();
      const sh = ctx.createRadialGradient(-thumbR*.2, -thumbR*.2, 0, 0, 0, thumbR);
      sh.addColorStop(0, 'rgba(255,255,255,.96)');
      sh.addColorStop(.5, 'rgba(255,255,255,.3)');
      sh.addColorStop(1, 'rgba(0,0,0,.07)');
      ctx.beginPath(); ctx.arc(0, 0, thumbR, 0, Math.PI * 2);
      ctx.fillStyle = sh; ctx.fill();
      ctx.restore();

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [W, H, size]);

  function handleClick() {
    const sw = swRef.current;
    sw.pressed = true;
    setTimeout(() => { sw.pressed = false; }, 80);
    sw.toggled = !sw.toggled;
    onChange?.(sw.toggled);
  }

  return (
    <canvas
      ref={cRef}
      width={W}
      height={H}
      style={{ cursor: 'pointer', display: 'block' }}
      onClick={handleClick}
    />
  );
}
