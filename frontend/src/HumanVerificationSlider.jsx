import { useEffect, useRef, useState } from 'react';
import { SliderPhysics } from './physics';

function getThemeValue(name, fallback) {
  if (typeof window === 'undefined') {
    return fallback;
  }
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export default function HumanVerificationSlider({ disabled = false, disabledText, onVerified }) {
  const cRef = useRef(null);
  const pRef = useRef(null);
  const rafRef = useRef(null);
  const tsRef = useRef(null);
  const drag = useRef(false);
  const verRef = useRef(false);
  const progRef = useRef(0);
  const [verified, setVerified] = useState(false);

  const W = 340;
  const H = 58;
  const TW = 52;
  const TH = 42;
  const PAD = 8;
  const TRACK = W - PAD * 2;
  const START_X = PAD + TW / 2;

  function createPhysics() {
    const physics = new SliderPhysics(10, TRACK);
    physics.init(PAD);
    physics.damping = 0.02;
    physics.setX(START_X);

    // Advance a couple of frames so the terminal point actually settles at the start.
    physics.update(1 / 60);
    physics.update(1 / 60);
    return physics;
  }

  useEffect(() => {
    pRef.current = createPhysics();
    progRef.current = 0;
    verRef.current = false;
    tsRef.current = null;
  }, [TRACK]);

  useEffect(() => {
    if (!disabled) {
      pRef.current = createPhysics();
      progRef.current = 0;
      verRef.current = false;
      tsRef.current = null;
      drag.current = false;
    }
  }, [disabled, TRACK]);

  useEffect(() => {
    if (verified || disabled) {
      return undefined;
    }

    const canvas = cRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) {
      return undefined;
    }

    function roundRect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.arcTo(x + w, y, x + w, y + r, r);
      ctx.lineTo(x + w, y + h - r);
      ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
      ctx.lineTo(x + r, y + h);
      ctx.arcTo(x, y + h, x, y + h - r, r);
      ctx.lineTo(x, y + r);
      ctx.arcTo(x, y, x + r, y, r);
      ctx.closePath();
    }

    function frame(ts) {
      if (!tsRef.current) {
        tsRef.current = ts;
      }

      const dt = Math.min((ts - tsRef.current) * 0.001, 0.033);
      tsRef.current = ts;
      pRef.current?.update(dt);
      ctx.clearRect(0, 0, W, H);

      const physics = pRef.current;
      if (!physics) {
        rafRef.current = requestAnimationFrame(frame);
        return;
      }

      const thumbX = physics.pts[physics.n - 1].x;
      const progress = Math.max(0, Math.min(1, (thumbX - PAD) / (TRACK - TW)));
      progRef.current = progress;
      const centerY = H / 2;
      const accent = getThemeValue('--accent', '#b24b7d');
      const accentLight = getThemeValue('--accent-light', 'rgba(178, 75, 125, 0.14)');
      const textSecondary = getThemeValue('--text-secondary', '#6b5058');
      const brandA = getThemeValue('--brand-a', '#d96c8d');
      const brandB = getThemeValue('--brand-b', '#b24b7d');
      const brandC = getThemeValue('--brand-c', '#9d76c8');
      const cardBg = getThemeValue('--card-bg', 'rgba(255, 250, 246, 0.9)');

      roundRect(PAD, centerY - 22, TRACK, 44, 14);
      ctx.fillStyle = cardBg;
      ctx.fill();

      if (progress > 0.01) {
        const fillWidth = Math.max(TW, (TRACK - TW) * progress + TW);
        roundRect(PAD, centerY - 22, fillWidth, 44, 14);
        ctx.fillStyle = accentLight.replace(/[\d.]+\)\s*$/, `${0.08 + progress * 0.22})`);
        ctx.fill();
      }

      const alpha = Math.max(0, 1 - progress * 2.5);
      if (alpha > 0) {
        ctx.save();
        ctx.font = "500 12px 'Sora', sans-serif";
        ctx.fillStyle = textSecondary.startsWith('#')
          ? `${textSecondary}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`
          : textSecondary;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Slide to verify ->', W / 2 + 28, centerY);
        ctx.restore();
      }

      const mid = physics.pts[Math.floor(physics.n / 2)].y;
      const scaleX = 1 + Math.abs(mid) * 0.025;
      const scaleY = Math.max(0.7, 1 - Math.abs(mid) * 0.018);
      ctx.save();
      ctx.translate(thumbX, centerY);
      ctx.scale(scaleX, scaleY);
      ctx.shadowColor = accent;
      ctx.shadowBlur = 14;
      ctx.shadowOffsetY = 3;
      roundRect(-TW / 2, -TH / 2, TW, TH, 11);
      const gradient = ctx.createLinearGradient(0, -TH / 2, 0, TH / 2);
      gradient.addColorStop(0, brandA);
      gradient.addColorStop(0.52, brandB);
      gradient.addColorStop(1, brandC);
      ctx.fillStyle = gradient;
      ctx.fill();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
      roundRect(-TW / 2, -TH / 2, TW, TH / 2, 11);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 17px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('->', 0, 0);
      ctx.restore();

      rafRef.current = requestAnimationFrame(frame);

      if (progress > 0.93 && !verRef.current) {
        verRef.current = true;
        cancelAnimationFrame(rafRef.current);
        setVerified(true);
        onVerified?.();
      }
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [disabled, onVerified, verified]);

  function getX(event) {
    const rect = cRef.current.getBoundingClientRect();
    const clientX = event.touches ? event.touches[0].clientX : event.clientX;
    return ((clientX - rect.left) * W) / rect.width;
  }

  function onDown(event) {
    if (disabled) {
      return;
    }
    drag.current = true;
    pRef.current?.setX(getX(event));
    event.preventDefault();
  }

  function onMove(event) {
    if (!drag.current || disabled) {
      return;
    }
    pRef.current?.setX(getX(event));
    event.preventDefault();
  }

  function onUp() {
    drag.current = false;
    if (progRef.current < 0.93) {
      pRef.current?.setX(START_X);
    }
  }

  if (verified) {
    return (
      <div className="verif-state verif-state-verified" style={{ width: W, height: H }}>
        <span>Human verified</span>
      </div>
    );
  }

  if (disabled) {
    return (
      <div className="verif-state verif-state-disabled" style={{ width: W, height: H }}>
        <span>{disabledText || 'Complete the required fields to enable verification.'}</span>
      </div>
    );
  }

  return (
    <canvas
      ref={cRef}
      width={W}
      height={H}
      style={{ cursor: 'pointer', touchAction: 'none', display: 'block', width: '100%', height: 'auto' }}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={onUp}
      onMouseLeave={onUp}
      onTouchStart={onDown}
      onTouchMove={onMove}
      onTouchEnd={onUp}
    />
  );
}
