import { useEffect } from 'react';
import { useSeasonal } from './context/SeasonalContext';
import { useTheme } from './context/ThemeContext';

const SPARKLE_ITEMS = Array.from({ length: 18 }, (_, index) => ({
  id: index,
  left: `${6 + (index * 5.2) % 88}%`,
  top: `${8 + (index * 13) % 78}%`,
  size: `${2 + (index % 5)}px`,
  floatDelay: `${(index % 7) * 0.6}s`,
  floatDuration: `${6 + (index % 7)}s`,
  twinkleDelay: `${(index % 6) * 0.35}s`,
  twinkleDuration: `${2 + (index % 5) * 0.45}s`,
  driftX: `${index % 2 === 0 ? 10 + (index % 4) * 5 : -(12 + (index % 4) * 4)}px`,
  driftY: `${-12 - (index % 5) * 4}px`,
}));

export default function SeasonalLayer() {
  const { mode, isEnabled, activeFestival, intensity, debugLabel } = useSeasonal();
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    console.info('[SeasonalLayer] render state', {
      mode,
      enabled: isEnabled,
      festival: activeFestival?.name ?? 'none',
      intensity,
      debugLabel,
      theme: resolvedTheme,
    });
  }, [mode, isEnabled, activeFestival, intensity, debugLabel, resolvedTheme]);

  if (!isEnabled || !activeFestival) {
    return null;
  }

  const style = {
    '--seasonal-glow-a': activeFestival.palette.glowA,
    '--seasonal-glow-b': activeFestival.palette.glowB,
    '--seasonal-glow-c': activeFestival.palette.glowC,
    '--seasonal-sparkle': activeFestival.palette.sparkle,
    '--seasonal-ornament': activeFestival.palette.ornament,
    '--seasonal-border': activeFestival.palette.border,
    '--seasonal-accent': activeFestival.accent,
  };

  return (
    <div
      className={`seasonal-layer seasonal-layer--${intensity} seasonal-layer--${resolvedTheme}`}
      style={style}
      aria-hidden="true"
    >
      <div className="seasonal-layer__warmth" />
      <div className="seasonal-layer__mesh" />
      <div className="seasonal-layer__glow seasonal-layer__glow--a" />
      <div className="seasonal-layer__glow seasonal-layer__glow--b" />
      <div className="seasonal-layer__glow seasonal-layer__glow--c" />

      <div className="seasonal-layer__corner seasonal-layer__corner--top-right" />
      <div className="seasonal-layer__corner seasonal-layer__corner--bottom-left" />
      <div className="seasonal-layer__diya seasonal-layer__diya--left" />
      <div className="seasonal-layer__diya seasonal-layer__diya--right" />

      <div className="seasonal-layer__sparkles">
        {SPARKLE_ITEMS.map((sparkle) => (
          <span
            key={sparkle.id}
            className="seasonal-layer__sparkle"
            style={{
              left: sparkle.left,
              top: sparkle.top,
              '--sparkle-size': sparkle.size,
              '--sparkle-float-delay': sparkle.floatDelay,
              '--sparkle-float-duration': sparkle.floatDuration,
              '--sparkle-twinkle-delay': sparkle.twinkleDelay,
              '--sparkle-twinkle-duration': sparkle.twinkleDuration,
              '--sparkle-drift-x': sparkle.driftX,
              '--sparkle-drift-y': sparkle.driftY,
            }}
          />
        ))}
      </div>
    </div>
  );
}
