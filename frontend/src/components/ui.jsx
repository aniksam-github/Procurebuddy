import { forwardRef } from 'react';

export function cn(...values) {
  return values.filter(Boolean).join(' ');
}

const BUTTON_VARIANTS = {
  primary:
    'bg-[color:var(--accent)] text-white shadow-[0_18px_45px_-24px_rgba(53,92,255,0.75)] hover:bg-[color:var(--accent-strong)]',
  secondary:
    'bg-[color:var(--card-strong)] text-[color:var(--text-primary)] border border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)]',
  ghost:
    'bg-transparent text-[color:var(--text-secondary)] hover:bg-[color:var(--card-subtle)] hover:text-[color:var(--text-primary)]',
};

const BUTTON_SIZES = {
  sm: 'h-10 px-4 text-sm',
  md: 'h-11 px-5 text-sm',
  lg: 'h-12 px-6 text-sm',
};

export const Button = forwardRef(function Button(
  { className = '', variant = 'primary', size = 'md', type = 'button', children, ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-2xl font-semibold transition duration-200',
        'disabled:cursor-not-allowed disabled:opacity-60',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
});

export const IconButton = forwardRef(function IconButton(
  { className = '', children, ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type="button"
      className={cn(
        'inline-flex h-10 w-10 items-center justify-center rounded-2xl',
        'border border-[color:var(--border-soft)] bg-[color:var(--card-strong)]',
        'text-[color:var(--text-secondary)] transition duration-200',
        'hover:border-[color:var(--border-strong)] hover:bg-[color:var(--card-hover)] hover:text-[color:var(--text-primary)]',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]',
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
});

export const Panel = forwardRef(function Panel(
  { as: Comp = 'div', className = '', children, ...props },
  ref
) {
  return (
    <Comp
      ref={ref}
      className={cn(
        'rounded-[28px] border border-[color:var(--border-soft)]',
        'bg-[color:var(--card-bg)] shadow-[var(--shadow-panel)]',
        'backdrop-blur-[var(--panel-blur)]',
        className
      )}
      {...props}
    >
      {children}
    </Comp>
  );
});

export function Eyebrow({ className = '', children }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em]',
        'border-[color:var(--border-strong)] bg-[color:var(--card-strong)] text-[color:var(--text-tertiary)]',
        className
      )}
    >
      {children}
    </span>
  );
}

export function Pill({ className = '', children }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium',
        'border-[color:var(--border-soft)] bg-[color:var(--card-strong)] text-[color:var(--text-secondary)]',
        className
      )}
    >
      {children}
    </span>
  );
}
