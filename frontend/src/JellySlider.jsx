import { useRef, useEffect } from 'react';
import { SliderPhysics } from './physics';

// ─────────────────────────────────────────────────
//  JellySlider.jsx
//  Canvas-rendered slider with soft-body XPBD physics
//  Props:
//    value      : 0–1
//    onChange   : (0–1) => void
//    color      : string hex
//    width      : number px (default 300)
//    height     : number px (default 52)
//    showValue  : bool
// ─────────────────────────────────────────────────

function getThemeValue(name, fallback) {
  if (typeof window === 'undefined') {
    return fallback;
  }
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export default function JellySlider({ value = 0.5, onChange, color, width = 300, height = 52, showValue = true }) {
  const cRef     = useRef(null);
  const pRef     = useRef(null);
  const rafRef   = useRef(null);
  const tsRef    = useRef(null);
  const drag     = useRef(false);
  const colorRef = useRef(color || getThemeValue('--accent', '#b24b7d'));
  const railRef = useRef(getThemeValue('--input-bg', '#dde1f0'));

  const PAD   = 22;
  const TRACK = width - PAD * 2;

  useEffect(() => {
    colorRef.current = color || getThemeValue('--accent', '#b24b7d');
    railRef.current = getThemeValue('--input-bg', '#dde1f0');
  }, [color]);

  useEffect(() => {
    const p = new SliderPhysics(13, TRACK);
    p.init(PAD);
    p.setX(PAD + value * TRACK);
    pRef.current = p;
  }, [TRACK, PAD]);

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
      const dt = Math.min((ts - tsRef.current) * 0.001, 0.033);
      tsRef.current = ts;
      pRef.current?.update(dt);
      ctx.clearRect(0, 0, width, height);

      const p = pRef.current;
      if (!p) { rafRef.current = requestAnimationFrame(frame); return; }

      const c = colorRef.current;
      const pts = p.pts, tX = pts[p.n-1].x, prog = (tX - PAD) / TRACK;
      const cy = height / 2, TH = 9;

      // Track
      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,.05)'; ctx.shadowBlur = 5; ctx.shadowOffsetY = 2;
      rr(PAD, cy - TH/2, TRACK, TH, TH/2);
      ctx.fillStyle = railRef.current; ctx.fill();
      ctx.restore();

      if (prog > 0.005) {
        const g = ctx.createLinearGradient(PAD, 0, tX, 0);
        g.addColorStop(0, c + '77'); g.addColorStop(1, c);
        rr(PAD, cy - TH/2, tX - PAD, TH, TH/2);
        ctx.fillStyle = g; ctx.fill();
      }

      // Jelly soft-body from physics particles
      if (pts.length > 1) {
        const bH = 8;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, cy - bH);
        for (let i = 1; i < pts.length; i++) {
          const mx = (pts[i-1].x + pts[i].x) / 2;
          ctx.quadraticCurveTo(mx, cy - bH - pts[i].y * 0.7, pts[i].x, cy - bH - pts[i].y * 0.4);
        }
        for (let i = pts.length-1; i >= 1; i--) {
          const mx = (pts[i-1].x + pts[i].x) / 2;
          ctx.quadraticCurveTo(mx, cy + bH + pts[i-1].y * 0.7, pts[i-1].x, cy + bH + pts[i-1].y * 0.4);
        }
        ctx.closePath();
        const jg = ctx.createLinearGradient(PAD, cy-20, tX, cy+20);
        jg.addColorStop(0, c + '33'); jg.addColorStop(0.5, c + 'aa'); jg.addColorStop(1, c + 'cc');
        ctx.fillStyle = jg; ctx.fill();
        const sg = ctx.createLinearGradient(PAD, cy-14, PAD, cy);
        sg.addColorStop(0, 'rgba(255,255,255,.38)'); sg.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = sg; ctx.fill();
      }

      // Thumb
      const TR = 12;
      ctx.save();
      ctx.shadowColor = c + '55'; ctx.shadowBlur = 12; ctx.shadowOffsetY = 3;
      ctx.beginPath(); ctx.arc(tX, cy, TR, 0, Math.PI * 2);
      ctx.fillStyle = 'white'; ctx.fill();
      const ts2 = ctx.createRadialGradient(tX-3, cy-4, 0, tX, cy, TR);
      ts2.addColorStop(0, 'rgba(255,255,255,1)');
      ts2.addColorStop(0.4, 'rgba(255,255,255,.55)');
      ts2.addColorStop(1, c + '22');
      ctx.beginPath(); ctx.arc(tX, cy, TR, 0, Math.PI * 2);
      ctx.fillStyle = ts2; ctx.fill();
      ctx.restore();

      if (showValue) {
        ctx.font = `600 10.5px 'Sora', sans-serif`;
        ctx.fillStyle = prog > 0.45 ? 'rgba(255,255,255,.85)' : c;
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(`${Math.round(prog * 100)}%`, tX - 15, cy + 1);
      }

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [width, height, PAD, TRACK, showValue]);

  function getX(e) {
    return (e.touches ? e.touches[0].clientX : e.clientX) - cRef.current.getBoundingClientRect().left;
  }
  function onDown(e) { drag.current = true; pRef.current?.setX(getX(e)); e.preventDefault(); }
  function onMove(e) {
    if (!drag.current) return;
    const x = getX(e);
    pRef.current?.setX(x);
    onChange?.((Math.max(PAD, Math.min(PAD + TRACK, x)) - PAD) / TRACK);
    e.preventDefault();
  }
  function onUp() { drag.current = false; }

  return (
    <canvas
      ref={cRef}
      width={width}
      height={height}
      style={{ cursor: 'pointer', touchAction: 'none', display: 'block' }}
      onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
      onTouchStart={onDown} onTouchMove={onMove} onTouchEnd={onUp}
    />
  );
}
