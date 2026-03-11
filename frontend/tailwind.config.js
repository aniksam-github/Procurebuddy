/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['DM Sans', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        ink: {
          50: '#f4f3f0',
          100: '#e8e6e0',
          200: '#d1cdc0',
          300: '#b5af9e',
          400: '#948d7a',
          500: '#787060',
          600: '#5f5850',
          700: '#4a4540',
          800: '#2e2b28',
          900: '#1a1815',
          950: '#0d0c0a',
        },
        amber: {
          400: '#fbbf24',
          500: '#f59e0b',
        },
      },
    },
  },
  plugins: [],
}