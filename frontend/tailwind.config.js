/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-body, Inter)', 'system-ui', 'sans-serif'],
        inter: ['Inter', 'system-ui', 'sans-serif'],
        satoshi: ['Satoshi', 'Inter', 'system-ui', 'sans-serif'],
        system: ['system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        primary: {
          50: '#eef4ff',
          100: '#dbe6ff',
          200: '#bfd1ff',
          300: '#96b2ff',
          400: '#668dff',
          500: '#355cff',
          600: '#2346e6',
          700: '#2038b8',
          800: '#21358f',
          900: '#1f316f',
        },
        ink: {
          50: '#f8fafc',
          100: '#e9eef5',
          200: '#d6deea',
          300: '#a6b2c6',
          400: '#6d7b96',
          500: '#49566f',
          600: '#334055',
          700: '#223045',
          800: '#142032',
          900: '#0b1526',
        },
        aura: {
          blue: '#8fc8ff',
          cyan: '#74d8e8',
          peach: '#ffd4bf',
        },
      },
      boxShadow: {
        panel: '0 32px 80px -40px rgba(16, 24, 40, 0.22)',
        soft: '0 20px 45px -30px rgba(15, 23, 42, 0.18)',
        glow: '0 20px 50px -26px rgba(53, 92, 255, 0.38)',
      },
      animation: {
        'pulse-soft': 'pulseSoft 3s ease-in-out infinite',
        'float-soft': 'floatSoft 6s ease-in-out infinite',
      },
      keyframes: {
        pulseSoft: {
          '0%, 100%': { opacity: '0.55' },
          '50%': { opacity: '1' },
        },
        floatSoft: {
          '0%, 100%': { transform: 'translate3d(0, 0, 0)' },
          '50%': { transform: 'translate3d(0, -10px, 0)' },
        },
      },
    },
  },
  plugins: [],
};
