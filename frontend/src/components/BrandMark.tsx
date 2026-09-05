interface BrandMarkProps {
  size?: number;
}

/**
 * RevenueGuard brand mark — custom inline SVG, no external assets.
 *
 * Symbolism: an open revenue-cycle ring (abstract "G" / rolling cycle) with a
 * clean upward-recovery arrow rising through its opening, signalling recovered
 * revenue and forward movement. Rendered with theme-aware colors (no box).
 */
export function BrandMark({ size = 32 }: BrandMarkProps) {
  return (
    <svg
      className="brand-mark"
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      focusable="false"
    >
      <path
        className="brand-mark__ring"
        d="M24 6 A18 18 0 1 0 42 24"
        stroke="currentColor"
        strokeWidth="4.2"
        strokeLinecap="round"
      />
      <path
        className="brand-mark__arrow"
        d="M9 37 L26 20"
        stroke="currentColor"
        strokeWidth="4.6"
        strokeLinecap="round"
      />
      <path
        className="brand-mark__arrow"
        d="M17 11 L26 20 L33 27"
        stroke="currentColor"
        strokeWidth="4.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * URL-encoded inline SVG favicon (same mark) for index.html. Kept here as a
 * single source of truth for manual updates.
 */
export const FAVICON_DATA_URI = [
  "data:image/svg+xml,",
  "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'%3E",
  "%3Crect width='48' height='48' rx='10' fill='%23111930'/%3E",
  "%3Cpath d='M24 6 A18 18 0 1 0 42 24' fill='none' stroke='%238b96ad' stroke-width='4.2' stroke-linecap='round'/%3E",
  "%3Cpath d='M9 37 L26 20' fill='none' stroke='%237aa5ff' stroke-width='4.6' stroke-linecap='round'/%3E",
  "%3Cpath d='M17 11 L26 20 L33 27' fill='none' stroke='%237aa5ff' stroke-width='4.6' stroke-linecap='round' stroke-linejoin='round'/%3E",
  "%3C/svg%3E",
].join("");