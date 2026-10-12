// Small stroke icons, 24x24, coloured by currentColor.
const props = {
  viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2,
  strokeLinecap: 'round', strokeLinejoin: 'round', className: 'icon', 'aria-hidden': true,
} as const;

export const UndoIcon = () => (
  <svg {...props}><path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></svg>
);

export const PlusIcon = () => (
  <svg {...props}><path d="M12 5v14M5 12h14" /></svg>
);

export const BagIcon = () => (
  <svg {...props}>
    <path d="M7 8h10l1.5 11.5a1 1 0 0 1-1 1.1h-11a1 1 0 0 1-1-1.1z" />
    <path d="M9 8V6.5a3 3 0 0 1 6 0V8" />
  </svg>
);

export const EyeIcon = () => (
  <svg {...props}>
    <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

export const GaugeIcon = () => (
  <svg {...props}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
);

export const CloseIcon = () => (
  <svg {...props}><path d="M6 6l12 12M18 6 6 18" /></svg>
);

export const RematchIcon = () => (
  <svg {...props}><path d="M3 12a9 9 0 0 1 15.5-6.2L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 0 1-15.5 6.2L3 16" /><path d="M3 21v-5h5" /></svg>
);

export const InfoIcon = () => (
  <svg {...props}><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5v.01" /></svg>
);

export const ChevronUpIcon = () => (
  <svg {...props}><path d="m6 15 6-6 6 6" /></svg>
);
