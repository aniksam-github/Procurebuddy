import { createContext, useContext, useEffect, useMemo, useState } from 'react';

const SEASONAL_MODE_KEY = 'procurebuddy-seasonal-mode';
const INDIA_TIME_ZONE = 'Asia/Kolkata';

const HARVEST_THEME = {
  id: 'harvest-season',
  name: 'Lohri / Makar Sankranti / Pongal / Magh Bihu',
  accent: '#f59e0b',
  palette: {
    glowA: 'rgba(56, 189, 248, 0.22)',
    glowB: 'rgba(245, 158, 11, 0.22)',
    glowC: 'rgba(250, 204, 21, 0.16)',
    sparkle: '#fde68a',
    ornament: 'rgba(245, 158, 11, 0.2)',
    border: 'rgba(245, 158, 11, 0.22)',
  },
};

const RAMADAN_THEME = {
  id: 'ramadan-roza',
  name: 'Ramadan / Roza month',
  accent: '#16a34a',
  palette: {
    glowA: 'rgba(22, 163, 74, 0.2)',
    glowB: 'rgba(14, 165, 233, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.15)',
    sparkle: '#dcfce7',
    ornament: 'rgba(22, 163, 74, 0.18)',
    border: 'rgba(34, 197, 94, 0.22)',
  },
};

const HOLI_THEME = {
  id: 'holi',
  name: 'Holi',
  accent: '#ec4899',
  palette: {
    glowA: 'rgba(236, 72, 153, 0.22)',
    glowB: 'rgba(34, 197, 94, 0.22)',
    glowC: 'rgba(59, 130, 246, 0.2)',
    sparkle: '#f9a8d4',
    ornament: 'rgba(236, 72, 153, 0.18)',
    border: 'rgba(99, 102, 241, 0.2)',
  },
};

const SPRING_NEW_YEAR_THEME = {
  id: 'spring-new-year',
  name: 'Chaitra Navratri / Ugadi / Gudi Padwa',
  accent: '#8b5cf6',
  palette: {
    glowA: 'rgba(139, 92, 246, 0.24)',
    glowB: 'rgba(236, 72, 153, 0.2)',
    glowC: 'rgba(250, 204, 21, 0.16)',
    sparkle: '#e9d5ff',
    ornament: 'rgba(139, 92, 246, 0.18)',
    border: 'rgba(168, 85, 247, 0.22)',
  },
};

const HOLY_WEEK_THEME = {
  id: 'holy-week',
  name: 'Holy Week / Good Friday / Easter',
  accent: '#6366f1',
  palette: {
    glowA: 'rgba(99, 102, 241, 0.22)',
    glowB: 'rgba(148, 163, 184, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.12)',
    sparkle: '#c7d2fe',
    ornament: 'rgba(99, 102, 241, 0.16)',
    border: 'rgba(99, 102, 241, 0.18)',
  },
};

const JAIN_THEME = {
  id: 'jain-observance',
  name: 'Mahavir Jayanti',
  accent: '#a16207',
  palette: {
    glowA: 'rgba(161, 98, 7, 0.22)',
    glowB: 'rgba(251, 191, 36, 0.18)',
    glowC: 'rgba(187, 247, 208, 0.14)',
    sparkle: '#fde68a',
    ornament: 'rgba(161, 98, 7, 0.18)',
    border: 'rgba(202, 138, 4, 0.2)',
  },
};

const BUDDHIST_THEME = {
  id: 'buddhist-observance',
  name: 'Buddha Purnima / Vesak',
  accent: '#0284c7',
  palette: {
    glowA: 'rgba(2, 132, 199, 0.22)',
    glowB: 'rgba(56, 189, 248, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.14)',
    sparkle: '#bae6fd',
    ornament: 'rgba(2, 132, 199, 0.18)',
    border: 'rgba(14, 165, 233, 0.2)',
  },
};

const ISLAMIC_NEW_YEAR_THEME = {
  id: 'muharram-ashura',
  name: 'Muharram / Ashura',
  accent: '#0f766e',
  palette: {
    glowA: 'rgba(15, 118, 110, 0.22)',
    glowB: 'rgba(6, 182, 212, 0.18)',
    glowC: 'rgba(148, 163, 184, 0.14)',
    sparkle: '#99f6e4',
    ornament: 'rgba(15, 118, 110, 0.18)',
    border: 'rgba(13, 148, 136, 0.2)',
  },
};

const MILAD_THEME = {
  id: 'milad-un-nabi',
  name: 'Milad un-Nabi / Eid-e-Milad',
  accent: '#0ea5e9',
  palette: {
    glowA: 'rgba(14, 165, 233, 0.22)',
    glowB: 'rgba(34, 197, 94, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.12)',
    sparkle: '#bae6fd',
    ornament: 'rgba(14, 165, 233, 0.18)',
    border: 'rgba(14, 165, 233, 0.2)',
  },
};

const PARSI_THEME = {
  id: 'parsi-new-year',
  name: 'Parsi New Year / Navroz',
  accent: '#14b8a6',
  palette: {
    glowA: 'rgba(20, 184, 166, 0.22)',
    glowB: 'rgba(56, 189, 248, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.12)',
    sparkle: '#99f6e4',
    ornament: 'rgba(20, 184, 166, 0.18)',
    border: 'rgba(20, 184, 166, 0.2)',
  },
};

const REGIONAL_NEW_YEAR_THEME = {
  id: 'regional-new-year',
  name: 'Baisakhi / Vishu / Bohag Bihu / Puthandu',
  accent: '#0f766e',
  palette: {
    glowA: 'rgba(15, 118, 110, 0.22)',
    glowB: 'rgba(245, 158, 11, 0.18)',
    glowC: 'rgba(34, 197, 94, 0.16)',
    sparkle: '#99f6e4',
    ornament: 'rgba(15, 118, 110, 0.18)',
    border: 'rgba(13, 148, 136, 0.2)',
  },
};

const EID_THEME = {
  id: 'eid',
  name: 'Eid season',
  accent: '#0891b2',
  palette: {
    glowA: 'rgba(8, 145, 178, 0.24)',
    glowB: 'rgba(22, 163, 74, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.14)',
    sparkle: '#bae6fd',
    ornament: 'rgba(8, 145, 178, 0.18)',
    border: 'rgba(6, 182, 212, 0.2)',
  },
};

const RAKSHA_BANDHAN_THEME = {
  id: 'raksha-bandhan',
  name: 'Raksha Bandhan',
  accent: '#ef4444',
  palette: {
    glowA: 'rgba(239, 68, 68, 0.22)',
    glowB: 'rgba(245, 158, 11, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.16)',
    sparkle: '#fde68a',
    ornament: 'rgba(234, 88, 12, 0.18)',
    border: 'rgba(245, 158, 11, 0.2)',
  },
};

const JANMASHTAMI_THEME = {
  id: 'janmashtami',
  name: 'Janmashtami',
  accent: '#2563eb',
  palette: {
    glowA: 'rgba(37, 99, 235, 0.22)',
    glowB: 'rgba(14, 165, 233, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.16)',
    sparkle: '#bfdbfe',
    ornament: 'rgba(37, 99, 235, 0.18)',
    border: 'rgba(59, 130, 246, 0.22)',
  },
};

const ONAM_THEME = {
  id: 'onam',
  name: 'Onam',
  accent: '#ca8a04',
  palette: {
    glowA: 'rgba(202, 138, 4, 0.24)',
    glowB: 'rgba(245, 158, 11, 0.2)',
    glowC: 'rgba(34, 197, 94, 0.16)',
    sparkle: '#fde68a',
    ornament: 'rgba(202, 138, 4, 0.18)',
    border: 'rgba(234, 179, 8, 0.22)',
  },
};

const GANESH_CHATURTHI_THEME = {
  id: 'ganesh-chaturthi',
  name: 'Ganesh Chaturthi',
  accent: '#ea580c',
  palette: {
    glowA: 'rgba(234, 88, 12, 0.24)',
    glowB: 'rgba(249, 115, 22, 0.22)',
    glowC: 'rgba(250, 204, 21, 0.18)',
    sparkle: '#fdba74',
    ornament: 'rgba(234, 88, 12, 0.2)',
    border: 'rgba(249, 115, 22, 0.22)',
  },
};

const NAVRATRI_THEME = {
  id: 'navratri-durga-puja-garba',
  name: 'Navratri / Durga Puja / Dandiya Nights',
  accent: '#f15b6c',
  palette: {
    glowA: 'rgba(244, 114, 182, 0.22)',
    glowB: 'rgba(249, 115, 22, 0.2)',
    glowC: 'rgba(251, 191, 36, 0.16)',
    sparkle: '#ffd66b',
    ornament: 'rgba(241, 91, 108, 0.18)',
    border: 'rgba(241, 91, 108, 0.2)',
  },
};

const DIWALI_THEME = {
  id: 'diwali-kali-puja-bandi-chhor',
  name: 'Diwali / Kali Puja / Bandi Chhor Divas',
  accent: '#f97316',
  palette: {
    glowA: 'rgba(249, 115, 22, 0.24)',
    glowB: 'rgba(250, 204, 21, 0.2)',
    glowC: 'rgba(244, 114, 182, 0.16)',
    sparkle: '#fde68a',
    ornament: 'rgba(249, 115, 22, 0.2)',
    border: 'rgba(251, 191, 36, 0.22)',
  },
};

const CHHATH_THEME = {
  id: 'chhath-puja',
  name: 'Chhath Puja',
  accent: '#f59e0b',
  palette: {
    glowA: 'rgba(245, 158, 11, 0.24)',
    glowB: 'rgba(249, 115, 22, 0.2)',
    glowC: 'rgba(34, 197, 94, 0.14)',
    sparkle: '#fde68a',
    ornament: 'rgba(245, 158, 11, 0.18)',
    border: 'rgba(249, 115, 22, 0.2)',
  },
};

const GURPURAB_THEME = {
  id: 'gurpurab',
  name: 'Guru Nanak Gurpurab',
  accent: '#7c3aed',
  palette: {
    glowA: 'rgba(124, 58, 237, 0.22)',
    glowB: 'rgba(59, 130, 246, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.14)',
    sparkle: '#ddd6fe',
    ornament: 'rgba(124, 58, 237, 0.18)',
    border: 'rgba(139, 92, 246, 0.2)',
  },
};

const CHRISTMAS_THEME = {
  id: 'christmas',
  name: 'Christmas',
  accent: '#16a34a',
  palette: {
    glowA: 'rgba(22, 163, 74, 0.2)',
    glowB: 'rgba(220, 38, 38, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.15)',
    sparkle: '#dcfce7',
    ornament: 'rgba(22, 163, 74, 0.18)',
    border: 'rgba(34, 197, 94, 0.22)',
  },
};

const CBRI_FOUNDATION_THEME = {
  id: 'cbri-foundation-day',
  name: 'CBRI Foundation Day',
  accent: '#1d4ed8',
  palette: {
    glowA: 'rgba(29, 78, 216, 0.24)',
    glowB: 'rgba(6, 182, 212, 0.18)',
    glowC: 'rgba(250, 204, 21, 0.14)',
    sparkle: '#bfdbfe',
    ornament: 'rgba(29, 78, 216, 0.18)',
    border: 'rgba(59, 130, 246, 0.22)',
  },
};

const SEASONAL_CSS_VARS = [
  '--seasonal-accent',
  '--seasonal-glow-a',
  '--seasonal-glow-b',
  '--seasonal-glow-c',
  '--seasonal-sparkle',
  '--seasonal-ornament',
  '--seasonal-border',
];

const SeasonalContext = createContext(null);

function getStoredMode() {
  if (typeof window === 'undefined') {
    return 'auto';
  }

  const value = window.localStorage.getItem(SEASONAL_MODE_KEY);
  if (value === 'on') {
    return 'always';
  }
  return value === 'always' || value === 'off' || value === 'auto' ? value : 'auto';
}

function shiftDateKey(dateKey, dayOffset) {
  const [year, month, day] = dateKey.split('-').map(Number);
  const shifted = new Date(Date.UTC(year, month - 1, day, 12));
  shifted.setUTCDate(shifted.getUTCDate() + dayOffset);
  return shifted.toISOString().slice(0, 10);
}

function createFestivalEvent({
  id,
  theme,
  startDate,
  endDate = startDate,
  leadDays = 3,
  trailDays = 0,
  priority = 50,
  noticeTitle,
  noticeBody,
}) {
  return {
    id,
    theme,
    startDate,
    endDate,
    effectiveStartDate: shiftDateKey(startDate, -leadDays),
    effectiveEndDate: shiftDateKey(endDate, trailDays),
    priority,
    noticeTitle,
    noticeBody,
  };
}

function createAnnualFoundationDayEvents(startYear, endYear) {
  const events = [];
  for (let year = startYear; year <= endYear; year += 1) {
    events.push(
      createFestivalEvent({
        id: `cbri-foundation-${year}`,
        theme: CBRI_FOUNDATION_THEME,
        startDate: `${year}-02-10`,
        priority: 95,
        noticeTitle: 'CBRI Foundation Day is today.',
        noticeBody:
          'Today marks CBRI Foundation Day, honoring the institute’s foundation in 1951 and its shared community spirit.',
      })
    );
  }
  return events;
}

const FESTIVAL_EVENTS = [
  ...createAnnualFoundationDayEvents(2025, 2035),
  createFestivalEvent({
    id: 'harvest-2025',
    theme: HARVEST_THEME,
    startDate: '2025-01-14',
    endDate: '2025-01-16',
    priority: 48,
    noticeTitle: 'Harvest celebrations start today.',
    noticeBody: 'Lohri, Makar Sankranti, Pongal, and Magh Bihu are beginning across communities today.',
  }),
  createFestivalEvent({
    id: 'ramadan-2025',
    theme: RAMADAN_THEME,
    startDate: '2025-03-02',
    endDate: '2025-03-30',
    priority: 70,
    noticeTitle: 'Ramadan / Roza starts today.',
    noticeBody: 'Ramadan begins today. Wishing everyone observing roza a peaceful and meaningful month.',
  }),
  createFestivalEvent({
    id: 'holi-2025',
    theme: HOLI_THEME,
    startDate: '2025-03-14',
    priority: 72,
    noticeTitle: 'Holi is today.',
    noticeBody: 'Holi celebrations begin today. Wishing everyone a bright, joyful, and colorful day.',
  }),
  createFestivalEvent({
    id: 'holy-week-2025',
    theme: HOLY_WEEK_THEME,
    startDate: '2025-04-18',
    endDate: '2025-04-20',
    priority: 66,
    noticeTitle: 'Good Friday is today.',
    noticeBody: 'Good Friday is being observed today, with Easter weekend following for Christian families and communities.',
  }),
  createFestivalEvent({
    id: 'mahavir-2025',
    theme: JAIN_THEME,
    startDate: '2025-04-10',
    priority: 67,
    noticeTitle: 'Mahavir Jayanti is today.',
    noticeBody: 'Mahavir Jayanti is being observed today. Wishing Jain colleagues and families peace, compassion, and harmony.',
  }),
  createFestivalEvent({
    id: 'regional-new-year-2025',
    theme: REGIONAL_NEW_YEAR_THEME,
    startDate: '2025-04-14',
    endDate: '2025-04-15',
    priority: 60,
    noticeTitle: 'Regional New Year celebrations start today.',
    noticeBody: 'Baisakhi, Vishu, Bohag Bihu, and Puthandu celebrations are starting across India today.',
  }),
  createFestivalEvent({
    id: 'buddha-purnima-2025',
    theme: BUDDHIST_THEME,
    startDate: '2025-05-12',
    priority: 65,
    noticeTitle: 'Buddha Purnima is today.',
    noticeBody: 'Buddha Purnima is being observed today. Wishing everyone reflecting on peace, compassion, and mindfulness a meaningful day.',
  }),
  createFestivalEvent({
    id: 'muharram-2025',
    theme: ISLAMIC_NEW_YEAR_THEME,
    startDate: '2025-07-06',
    priority: 64,
    noticeTitle: 'Muharram starts today.',
    noticeBody: 'Muharram begins today. Respectful wishes to everyone observing the Islamic New Year and Ashura remembrance.',
  }),
  createFestivalEvent({
    id: 'rakhi-2025',
    theme: RAKSHA_BANDHAN_THEME,
    startDate: '2025-08-09',
    priority: 58,
    noticeTitle: 'Raksha Bandhan is today.',
    noticeBody: 'Raksha Bandhan is being celebrated today. Wishing everyone warmth, care, and family joy.',
  }),
  createFestivalEvent({
    id: 'janmashtami-2025',
    theme: JANMASHTAMI_THEME,
    startDate: '2025-08-16',
    priority: 59,
    noticeTitle: 'Janmashtami is today.',
    noticeBody: 'Janmashtami celebrations begin today. Wishing everyone a peaceful and devotional festival.',
  }),
  createFestivalEvent({
    id: 'ganesh-2025',
    theme: GANESH_CHATURTHI_THEME,
    startDate: '2025-08-27',
    endDate: '2025-08-30',
    priority: 61,
    noticeTitle: 'Ganesh Chaturthi starts today.',
    noticeBody: 'Ganesh Chaturthi begins today. Wishing everyone a joyful and auspicious celebration.',
  }),
  createFestivalEvent({
    id: 'onam-2025',
    theme: ONAM_THEME,
    startDate: '2025-09-05',
    endDate: '2025-09-08',
    priority: 57,
    noticeTitle: 'Onam starts today.',
    noticeBody: 'Onam festivities begin today. Wishing everyone joy, togetherness, and abundance.',
  }),
  createFestivalEvent({
    id: 'parsi-new-year-2025',
    theme: PARSI_THEME,
    startDate: '2025-08-15',
    priority: 56,
    noticeTitle: 'Parsi New Year is today.',
    noticeBody: 'Navroz is being celebrated today. Best wishes for joy, renewal, and prosperity to everyone observing.',
  }),
  createFestivalEvent({
    id: 'milad-2025',
    theme: MILAD_THEME,
    startDate: '2025-09-05',
    priority: 69,
    noticeTitle: 'Milad un-Nabi is today.',
    noticeBody: 'Milad un-Nabi is being observed today. Respectful wishes to everyone commemorating the Prophet’s life and teachings.',
  }),
  createFestivalEvent({
    id: 'navratri-2025',
    theme: NAVRATRI_THEME,
    startDate: '2025-09-22',
    endDate: '2025-10-02',
    priority: 80,
    noticeTitle: 'Navratri season starts today.',
    noticeBody: 'Sharad Navratri begins today, with Durga Puja and dandiya-garba celebrations unfolding across the week.',
  }),
  createFestivalEvent({
    id: 'diwali-2025',
    theme: DIWALI_THEME,
    startDate: '2025-10-20',
    endDate: '2025-10-24',
    priority: 90,
    noticeTitle: 'Diwali season starts today.',
    noticeBody: 'Diwali, Kali Puja, and Bandi Chhor Divas observances are beginning today across communities.',
  }),
  createFestivalEvent({
    id: 'chhath-2025',
    theme: CHHATH_THEME,
    startDate: '2025-10-28',
    endDate: '2025-10-31',
    priority: 63,
    noticeTitle: 'Chhath Puja starts today.',
    noticeBody: 'Chhath Puja observances begin today. Wishing everyone strength, peace, and devotion.',
  }),
  createFestivalEvent({
    id: 'gurpurab-2025',
    theme: GURPURAB_THEME,
    startDate: '2025-11-05',
    priority: 62,
    noticeTitle: 'Guru Nanak Gurpurab is today.',
    noticeBody: 'Guru Nanak Gurpurab is being observed today. Wishing everyone peace, seva, and light.',
  }),
  createFestivalEvent({
    id: 'christmas-2025',
    theme: CHRISTMAS_THEME,
    startDate: '2025-12-25',
    endDate: '2025-12-26',
    priority: 68,
    noticeTitle: 'Christmas is today.',
    noticeBody: 'Christmas is being celebrated today. Wishing everyone joy, peace, and togetherness.',
  }),
  createFestivalEvent({
    id: 'harvest-2026',
    theme: HARVEST_THEME,
    startDate: '2026-01-14',
    endDate: '2026-01-16',
    priority: 48,
    noticeTitle: 'Harvest celebrations start today.',
    noticeBody: 'Lohri, Makar Sankranti, Pongal, and Magh Bihu are beginning across communities today.',
  }),
  createFestivalEvent({
    id: 'ramadan-2026',
    theme: RAMADAN_THEME,
    startDate: '2026-02-19',
    endDate: '2026-03-20',
    priority: 70,
    noticeTitle: 'Ramadan / Roza starts today.',
    noticeBody: 'Ramadan begins today. Wishing everyone observing roza a peaceful and meaningful month.',
  }),
  createFestivalEvent({
    id: 'holi-2026',
    theme: HOLI_THEME,
    startDate: '2026-03-04',
    priority: 72,
    noticeTitle: 'Holi is today.',
    noticeBody: 'Holi celebrations begin today. Wishing everyone a bright, joyful, and colorful day.',
  }),
  createFestivalEvent({
    id: 'spring-new-year-2026',
    theme: SPRING_NEW_YEAR_THEME,
    startDate: '2026-03-19',
    endDate: '2026-03-27',
    priority: 74,
    noticeTitle: 'Chaitra Navratri season starts today.',
    noticeBody: 'Chaitra Navratri begins today, with Ugadi and Gudi Padwa celebrations close alongside it.',
  }),
  createFestivalEvent({
    id: 'eid-fitr-2026',
    theme: EID_THEME,
    startDate: '2026-03-21',
    endDate: '2026-03-22',
    priority: 76,
    noticeTitle: 'Eid al-Fitr is today.',
    noticeBody: 'Eid al-Fitr is being celebrated today. Eid Mubarak to everyone observing.',
  }),
  createFestivalEvent({
    id: 'mahavir-2026',
    theme: JAIN_THEME,
    startDate: '2026-03-31',
    priority: 67,
    noticeTitle: 'Mahavir Jayanti is today.',
    noticeBody: 'Mahavir Jayanti is being observed today. Wishing Jain colleagues and families peace, compassion, and harmony.',
  }),
  createFestivalEvent({
    id: 'holy-week-2026',
    theme: HOLY_WEEK_THEME,
    startDate: '2026-04-03',
    endDate: '2026-04-05',
    priority: 66,
    noticeTitle: 'Good Friday is today.',
    noticeBody: 'Good Friday is being observed today, with Easter weekend following for Christian families and communities.',
  }),
  createFestivalEvent({
    id: 'regional-new-year-2026',
    theme: REGIONAL_NEW_YEAR_THEME,
    startDate: '2026-04-14',
    endDate: '2026-04-15',
    priority: 60,
    noticeTitle: 'Regional New Year celebrations start today.',
    noticeBody: 'Baisakhi, Vishu, Bohag Bihu, and Puthandu celebrations are starting across India today.',
  }),
  createFestivalEvent({
    id: 'buddha-purnima-2026',
    theme: BUDDHIST_THEME,
    startDate: '2026-05-01',
    priority: 65,
    noticeTitle: 'Buddha Purnima is today.',
    noticeBody: 'Buddha Purnima is being observed today. Wishing everyone reflecting on peace, compassion, and mindfulness a meaningful day.',
  }),
  createFestivalEvent({
    id: 'bakrid-2026',
    theme: EID_THEME,
    startDate: '2026-05-27',
    endDate: '2026-05-28',
    priority: 64,
    noticeTitle: 'Bakrid is today.',
    noticeBody: 'Eid al-Adha is being observed today. Wishing everyone peace, generosity, and togetherness.',
  }),
  createFestivalEvent({
    id: 'muharram-2026',
    theme: ISLAMIC_NEW_YEAR_THEME,
    startDate: '2026-06-26',
    priority: 64,
    noticeTitle: 'Muharram starts today.',
    noticeBody: 'Muharram begins today. Respectful wishes to everyone observing the Islamic New Year and Ashura remembrance.',
  }),
  createFestivalEvent({
    id: 'onam-2026',
    theme: ONAM_THEME,
    startDate: '2026-08-26',
    endDate: '2026-08-29',
    priority: 57,
    noticeTitle: 'Onam starts today.',
    noticeBody: 'Onam festivities begin today. Wishing everyone joy, togetherness, and abundance.',
  }),
  createFestivalEvent({
    id: 'milad-2026',
    theme: MILAD_THEME,
    startDate: '2026-08-26',
    priority: 69,
    noticeTitle: 'Milad un-Nabi is today.',
    noticeBody: 'Milad un-Nabi is being observed today. Respectful wishes to everyone commemorating the Prophet’s life and teachings.',
  }),
  createFestivalEvent({
    id: 'parsi-new-year-2026',
    theme: PARSI_THEME,
    startDate: '2026-08-15',
    priority: 56,
    noticeTitle: 'Parsi New Year is today.',
    noticeBody: 'Navroz is being celebrated today. Best wishes for joy, renewal, and prosperity to everyone observing.',
  }),
  createFestivalEvent({
    id: 'rakhi-2026',
    theme: RAKSHA_BANDHAN_THEME,
    startDate: '2026-08-28',
    priority: 58,
    noticeTitle: 'Raksha Bandhan is today.',
    noticeBody: 'Raksha Bandhan is being celebrated today. Wishing everyone warmth, care, and family joy.',
  }),
  createFestivalEvent({
    id: 'janmashtami-2026',
    theme: JANMASHTAMI_THEME,
    startDate: '2026-09-04',
    priority: 59,
    noticeTitle: 'Janmashtami is today.',
    noticeBody: 'Janmashtami celebrations begin today. Wishing everyone a peaceful and devotional festival.',
  }),
  createFestivalEvent({
    id: 'ganesh-2026',
    theme: GANESH_CHATURTHI_THEME,
    startDate: '2026-09-14',
    endDate: '2026-09-17',
    priority: 61,
    noticeTitle: 'Ganesh Chaturthi starts today.',
    noticeBody: 'Ganesh Chaturthi begins today. Wishing everyone a joyful and auspicious celebration.',
  }),
  createFestivalEvent({
    id: 'navratri-2026',
    theme: NAVRATRI_THEME,
    startDate: '2026-10-11',
    endDate: '2026-10-20',
    priority: 80,
    noticeTitle: 'Navratri season starts today.',
    noticeBody: 'Sharad Navratri begins today, with Durga Puja and dandiya-garba celebrations unfolding across the week.',
  }),
  createFestivalEvent({
    id: 'diwali-2026',
    theme: DIWALI_THEME,
    startDate: '2026-11-08',
    endDate: '2026-11-12',
    priority: 90,
    noticeTitle: 'Diwali season starts today.',
    noticeBody: 'Diwali, Kali Puja, and Bandi Chhor Divas observances are beginning today across communities.',
  }),
  createFestivalEvent({
    id: 'chhath-2026',
    theme: CHHATH_THEME,
    startDate: '2026-11-15',
    endDate: '2026-11-18',
    priority: 63,
    noticeTitle: 'Chhath Puja starts today.',
    noticeBody: 'Chhath Puja observances begin today. Wishing everyone strength, peace, and devotion.',
  }),
  createFestivalEvent({
    id: 'gurpurab-2026',
    theme: GURPURAB_THEME,
    startDate: '2026-11-24',
    priority: 62,
    noticeTitle: 'Guru Nanak Gurpurab is today.',
    noticeBody: 'Guru Nanak Gurpurab is being observed today. Wishing everyone peace, seva, and light.',
  }),
  createFestivalEvent({
    id: 'christmas-2026',
    theme: CHRISTMAS_THEME,
    startDate: '2026-12-25',
    endDate: '2026-12-26',
    priority: 68,
    noticeTitle: 'Christmas is today.',
    noticeBody: 'Christmas is being celebrated today. Wishing everyone joy, peace, and togetherness.',
  }),
  createFestivalEvent({
    id: 'harvest-2027',
    theme: HARVEST_THEME,
    startDate: '2027-01-15',
    endDate: '2027-01-17',
    priority: 48,
    noticeTitle: 'Harvest celebrations start today.',
    noticeBody: 'Lohri, Makar Sankranti, Pongal, and Magh Bihu are beginning across communities today.',
  }),
  createFestivalEvent({
    id: 'ramadan-2027',
    theme: RAMADAN_THEME,
    startDate: '2027-02-09',
    endDate: '2027-03-10',
    priority: 70,
    noticeTitle: 'Ramadan / Roza starts today.',
    noticeBody: 'Ramadan begins today. Wishing everyone observing roza a peaceful and meaningful month.',
  }),
  createFestivalEvent({
    id: 'holi-2027',
    theme: HOLI_THEME,
    startDate: '2027-03-22',
    priority: 72,
    noticeTitle: 'Holi is today.',
    noticeBody: 'Holi celebrations begin today. Wishing everyone a bright, joyful, and colorful day.',
  }),
  createFestivalEvent({
    id: 'holy-week-2027',
    theme: HOLY_WEEK_THEME,
    startDate: '2027-03-26',
    endDate: '2027-03-28',
    priority: 66,
    noticeTitle: 'Good Friday is today.',
    noticeBody: 'Good Friday is being observed today, with Easter weekend following for Christian families and communities.',
  }),
  createFestivalEvent({
    id: 'spring-new-year-2027',
    theme: SPRING_NEW_YEAR_THEME,
    startDate: '2027-04-07',
    endDate: '2027-04-15',
    priority: 74,
    noticeTitle: 'Chaitra Navratri season starts today.',
    noticeBody: 'Chaitra Navratri begins today, with Ugadi and Gudi Padwa celebrations close alongside it.',
  }),
  createFestivalEvent({
    id: 'regional-new-year-2027',
    theme: REGIONAL_NEW_YEAR_THEME,
    startDate: '2027-04-14',
    endDate: '2027-04-15',
    priority: 60,
    noticeTitle: 'Regional New Year celebrations start today.',
    noticeBody: 'Baisakhi, Vishu, Bohag Bihu, and Puthandu celebrations are starting across India today.',
  }),
  createFestivalEvent({
    id: 'mahavir-2027',
    theme: JAIN_THEME,
    startDate: '2027-04-19',
    priority: 67,
    noticeTitle: 'Mahavir Jayanti is today.',
    noticeBody: 'Mahavir Jayanti is being observed today. Wishing Jain colleagues and families peace, compassion, and harmony.',
  }),
  createFestivalEvent({
    id: 'bakrid-2027',
    theme: EID_THEME,
    startDate: '2027-05-17',
    endDate: '2027-05-18',
    priority: 64,
    noticeTitle: 'Bakrid is today.',
    noticeBody: 'Eid al-Adha is being observed today. Wishing everyone peace, generosity, and togetherness.',
  }),
  createFestivalEvent({
    id: 'buddha-purnima-2027',
    theme: BUDDHIST_THEME,
    startDate: '2027-05-20',
    priority: 65,
    noticeTitle: 'Buddha Purnima is today.',
    noticeBody: 'Buddha Purnima is being observed today. Wishing everyone reflecting on peace, compassion, and mindfulness a meaningful day.',
  }),
  createFestivalEvent({
    id: 'muharram-2027',
    theme: ISLAMIC_NEW_YEAR_THEME,
    startDate: '2027-06-16',
    priority: 64,
    noticeTitle: 'Muharram starts today.',
    noticeBody: 'Muharram begins today. Respectful wishes to everyone observing the Islamic New Year and Ashura remembrance.',
  }),
  createFestivalEvent({
    id: 'rakhi-2027',
    theme: RAKSHA_BANDHAN_THEME,
    startDate: '2027-08-17',
    priority: 58,
    noticeTitle: 'Raksha Bandhan is today.',
    noticeBody: 'Raksha Bandhan is being celebrated today. Wishing everyone warmth, care, and family joy.',
  }),
  createFestivalEvent({
    id: 'parsi-new-year-2027',
    theme: PARSI_THEME,
    startDate: '2027-08-15',
    priority: 56,
    noticeTitle: 'Parsi New Year is today.',
    noticeBody: 'Navroz is being celebrated today. Best wishes for joy, renewal, and prosperity to everyone observing.',
  }),
  createFestivalEvent({
    id: 'milad-2027',
    theme: MILAD_THEME,
    startDate: '2027-08-15',
    priority: 69,
    noticeTitle: 'Milad un-Nabi is today.',
    noticeBody: 'Milad un-Nabi is being observed today. Respectful wishes to everyone commemorating the Prophet’s life and teachings.',
  }),
  createFestivalEvent({
    id: 'janmashtami-2027',
    theme: JANMASHTAMI_THEME,
    startDate: '2027-08-25',
    priority: 59,
    noticeTitle: 'Janmashtami is today.',
    noticeBody: 'Janmashtami celebrations begin today. Wishing everyone a peaceful and devotional festival.',
  }),
  createFestivalEvent({
    id: 'ganesh-2027',
    theme: GANESH_CHATURTHI_THEME,
    startDate: '2027-09-04',
    endDate: '2027-09-07',
    priority: 61,
    noticeTitle: 'Ganesh Chaturthi starts today.',
    noticeBody: 'Ganesh Chaturthi begins today. Wishing everyone a joyful and auspicious celebration.',
  }),
  createFestivalEvent({
    id: 'onam-2027',
    theme: ONAM_THEME,
    startDate: '2027-09-12',
    endDate: '2027-09-15',
    priority: 57,
    noticeTitle: 'Onam starts today.',
    noticeBody: 'Onam festivities begin today. Wishing everyone joy, togetherness, and abundance.',
  }),
  createFestivalEvent({
    id: 'navratri-2027',
    theme: NAVRATRI_THEME,
    startDate: '2027-09-30',
    endDate: '2027-10-09',
    priority: 80,
    noticeTitle: 'Navratri season starts today.',
    noticeBody: 'Sharad Navratri begins today, with Durga Puja and dandiya-garba celebrations unfolding across the week.',
  }),
  createFestivalEvent({
    id: 'diwali-2027',
    theme: DIWALI_THEME,
    startDate: '2027-10-29',
    endDate: '2027-11-02',
    priority: 90,
    noticeTitle: 'Diwali season starts today.',
    noticeBody: 'Diwali, Kali Puja, and Bandi Chhor Divas observances are beginning today across communities.',
  }),
  createFestivalEvent({
    id: 'chhath-2027',
    theme: CHHATH_THEME,
    startDate: '2027-11-04',
    endDate: '2027-11-07',
    priority: 63,
    noticeTitle: 'Chhath Puja starts today.',
    noticeBody: 'Chhath Puja observances begin today. Wishing everyone strength, peace, and devotion.',
  }),
  createFestivalEvent({
    id: 'christmas-2027',
    theme: CHRISTMAS_THEME,
    startDate: '2027-12-25',
    endDate: '2027-12-26',
    priority: 68,
    noticeTitle: 'Christmas is today.',
    noticeBody: 'Christmas is being celebrated today. Wishing everyone joy, peace, and togetherness.',
  }),
].sort((left, right) => left.effectiveStartDate.localeCompare(right.effectiveStartDate));

function getIndiaDateKey(date = new Date()) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: INDIA_TIME_ZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);

  const year = parts.find((part) => part.type === 'year')?.value ?? '0000';
  const month = parts.find((part) => part.type === 'month')?.value ?? '01';
  const day = parts.find((part) => part.type === 'day')?.value ?? '01';

  return `${year}-${month}-${day}`;
}

function formatDateLabel(dateKey) {
  const [year, month, day] = dateKey.split('-').map(Number);
  return new Intl.DateTimeFormat('en-IN', {
    timeZone: INDIA_TIME_ZONE,
    month: 'short',
    day: 'numeric',
  }).format(new Date(Date.UTC(year, month - 1, day, 12)));
}

function getActiveEvents(date = new Date()) {
  const today = getIndiaDateKey(date);
  return FESTIVAL_EVENTS.filter((event) => today >= event.effectiveStartDate && today <= event.effectiveEndDate).sort(
    (left, right) => right.priority - left.priority || right.startDate.localeCompare(left.startDate)
  );
}

function getUpcomingFestival(date = new Date()) {
  const today = getIndiaDateKey(date);
  return FESTIVAL_EVENTS.find((event) => event.startDate > today) ?? null;
}

function getTodayAnnouncements(date = new Date()) {
  const today = getIndiaDateKey(date);
  return FESTIVAL_EVENTS.filter((event) => event.startDate === today).sort(
    (left, right) => right.priority - left.priority || left.theme.name.localeCompare(right.theme.name)
  );
}

function buildAnnouncement(todayAnnouncements) {
  if (todayAnnouncements.length === 0) {
    return null;
  }

  if (todayAnnouncements.length === 1) {
    const event = todayAnnouncements[0];
    return {
      id: `${event.id}:${event.startDate}`,
      title: event.noticeTitle,
      messages: [event.noticeBody],
    };
  }

  return {
    id: `${todayAnnouncements.map((event) => event.id).join('|')}:${todayAnnouncements[0].startDate}`,
    title: 'Observances starting today',
    messages: todayAnnouncements.map((event) => event.noticeBody),
  };
}

function resolveSeasonalState(mode, date = new Date()) {
  const activeEvents = getActiveEvents(date);
  const activeEvent = activeEvents[0] ?? null;
  const upcomingFestival = getUpcomingFestival(date);
  const todayAnnouncements = getTodayAnnouncements(date);

  if (mode === 'off') {
    return {
      mode,
      isEnabled: false,
      activeFestival: null,
      upcomingFestival,
      upcomingFestivalLabel: upcomingFestival
        ? `${upcomingFestival.theme.name} on ${formatDateLabel(upcomingFestival.startDate)}`
        : '',
      announcement: null,
      intensity: 'off',
      debugLabel: 'Seasonal layer disabled',
    };
  }

  if (mode === 'always') {
    return {
      mode: 'always',
      isEnabled: true,
      activeFestival: activeEvent?.theme ?? DIWALI_THEME,
      upcomingFestival,
      upcomingFestivalLabel: upcomingFestival
        ? `${upcomingFestival.theme.name} on ${formatDateLabel(upcomingFestival.startDate)}`
        : '',
      announcement: buildAnnouncement(todayAnnouncements),
      intensity: 'strong',
      debugLabel: activeEvent
        ? `Seasonal layer forced on during ${activeEvent.theme.name}`
        : 'Seasonal layer forced on',
    };
  }

  return {
    mode: 'auto',
    isEnabled: Boolean(activeEvent),
    activeFestival: activeEvent?.theme ?? null,
    upcomingFestival,
    upcomingFestivalLabel: upcomingFestival
      ? `${upcomingFestival.theme.name} on ${formatDateLabel(upcomingFestival.startDate)}`
      : '',
    announcement: buildAnnouncement(todayAnnouncements),
    intensity: activeEvent ? 'soft' : 'off',
    debugLabel: activeEvent
      ? `Auto-detected festival: ${activeEvent.theme.name}`
      : upcomingFestival
        ? `Next festival: ${upcomingFestival.theme.name}`
        : 'No active seasonal festival',
  };
}

export function SeasonalProvider({ children }) {
  const [mode, setModeState] = useState(getStoredMode);
  const [now, setNow] = useState(() => Date.now());
  const resolved = useMemo(() => resolveSeasonalState(mode, new Date(now)), [mode, now]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setNow(Date.now());
    }, 30 * 60 * 1000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    window.localStorage.setItem(SEASONAL_MODE_KEY, mode);
    root.dataset.seasonalMode = mode;
    root.dataset.seasonalFestival = resolved.activeFestival?.id || 'none';
    root.dataset.seasonalActive = resolved.isEnabled ? 'true' : 'false';

    if (resolved.activeFestival) {
      root.style.setProperty('--seasonal-accent', resolved.activeFestival.accent);
      root.style.setProperty('--seasonal-glow-a', resolved.activeFestival.palette.glowA);
      root.style.setProperty('--seasonal-glow-b', resolved.activeFestival.palette.glowB);
      root.style.setProperty('--seasonal-glow-c', resolved.activeFestival.palette.glowC);
      root.style.setProperty('--seasonal-sparkle', resolved.activeFestival.palette.sparkle);
      root.style.setProperty('--seasonal-ornament', resolved.activeFestival.palette.ornament);
      root.style.setProperty('--seasonal-border', resolved.activeFestival.palette.border);
    } else {
      SEASONAL_CSS_VARS.forEach((cssVar) => root.style.removeProperty(cssVar));
    }

    console.info('[SeasonalProvider] mode changed', {
      mode,
      enabled: resolved.isEnabled,
      festival: resolved.activeFestival?.name ?? 'none',
      nextFestival: resolved.upcomingFestival?.theme.name ?? 'none',
      announcement: resolved.announcement?.title ?? 'none',
      intensity: resolved.intensity,
    });
  }, [mode, resolved]);

  function setMode(nextMode) {
    const normalizedMode = nextMode === 'on' ? 'always' : nextMode;
    console.info('[SeasonalProvider] setMode called', { nextMode: normalizedMode });
    setModeState(normalizedMode);
  }

  const value = useMemo(
    () => ({
      mode,
      setMode,
      isEnabled: resolved.isEnabled,
      activeFestival: resolved.activeFestival,
      upcomingFestival: resolved.upcomingFestival,
      upcomingFestivalLabel: resolved.upcomingFestivalLabel,
      announcement: resolved.announcement,
      intensity: resolved.intensity,
      debugLabel: resolved.debugLabel,
    }),
    [mode, resolved]
  );

  return <SeasonalContext.Provider value={value}>{children}</SeasonalContext.Provider>;
}

export function useSeasonal() {
  const context = useContext(SeasonalContext);
  if (!context) {
    throw new Error('useSeasonal must be used within SeasonalProvider');
  }
  return context;
}
