const FESTIVAL_CALENDAR = {
  2026: [
    {
      id: 'holi',
      name: 'Holi',
      leadDays: 7,
      start: '2026-03-03T00:00:00+05:30',
      end: '2026-03-05T23:59:59+05:30',
      palette: {
        accent: '#f97316',
        brand: ['#38bdf8', '#f97316', '#ec4899', '#a855f7'],
        scene: ['rgba(56, 189, 248, 0.24)', 'rgba(249, 115, 22, 0.22)', 'rgba(236, 72, 153, 0.2)'],
      },
      effect: 'holi',
    },
    {
      id: 'ramadan',
      name: 'Ramadan / Eid',
      leadDays: 7,
      start: '2026-03-19T00:00:00+05:30',
      end: '2026-03-22T23:59:59+05:30',
      palette: {
        accent: '#0f766e',
        brand: ['#22c55e', '#14b8a6', '#0ea5e9', '#f8fafc'],
        scene: ['rgba(15, 118, 110, 0.18)', 'rgba(14, 165, 233, 0.14)', 'rgba(248, 250, 252, 0.08)'],
      },
      effect: 'ramadan',
    },
    {
      id: 'navratri',
      name: 'Navratri',
      leadDays: 7,
      start: '2026-03-19T00:00:00+05:30',
      end: '2026-03-27T23:59:59+05:30',
      palette: {
        accent: '#dc2626',
        brand: ['#fb7185', '#ef4444', '#f59e0b', '#facc15'],
        scene: ['rgba(239, 68, 68, 0.2)', 'rgba(245, 158, 11, 0.16)', 'rgba(250, 204, 21, 0.12)'],
      },
      effect: 'navratri',
    },
    {
      id: 'durgapuja',
      name: 'Durga Puja',
      leadDays: 7,
      start: '2026-10-17T00:00:00+05:30',
      end: '2026-10-20T23:59:59+05:30',
      palette: {
        accent: '#ea580c',
        brand: ['#fb7185', '#ea580c', '#f59e0b', '#fde68a'],
        scene: ['rgba(251, 113, 133, 0.18)', 'rgba(234, 88, 12, 0.18)', 'rgba(253, 230, 138, 0.16)'],
      },
      effect: 'durgapuja',
    },
    {
      id: 'diwali',
      name: 'Diwali',
      leadDays: 7,
      start: '2026-11-06T00:00:00+05:30',
      end: '2026-11-10T23:59:59+05:30',
      palette: {
        accent: '#f59e0b',
        brand: ['#fb7185', '#f97316', '#f59e0b', '#fde047'],
        scene: ['rgba(249, 115, 22, 0.18)', 'rgba(245, 158, 11, 0.18)', 'rgba(253, 224, 71, 0.16)'],
      },
      effect: 'diwali',
    },
    {
      id: 'christmas',
      name: 'Christmas',
      leadDays: 7,
      start: '2026-12-20T00:00:00+05:30',
      end: '2026-12-27T23:59:59+05:30',
      palette: {
        accent: '#dc2626',
        brand: ['#22c55e', '#ef4444', '#f8fafc', '#93c5fd'],
        scene: ['rgba(34, 197, 94, 0.14)', 'rgba(239, 68, 68, 0.18)', 'rgba(147, 197, 253, 0.18)'],
      },
      effect: 'christmas',
    },
  ],
};

function getFestivalEntries(date) {
  const yearEntries = FESTIVAL_CALENDAR[date.getFullYear()] || [];
  return yearEntries.filter((festival) => {
    const start = new Date(festival.start);
    start.setDate(start.getDate() - (festival.leadDays ?? 0));
    const end = new Date(festival.end);
    return date >= start && date <= end;
  });
}

export function getFestivalContext(date, festiveMode = 'auto') {
  if (festiveMode === 'off') {
    return null;
  }

  const activeFestivals = getFestivalEntries(date);
  if (!activeFestivals.length) {
    return null;
  }

  const index = Math.floor(date.getHours() / 6) % activeFestivals.length;
  return activeFestivals[index];
}
