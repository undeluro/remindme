// Kinly logo: a pair of glasses (the product's end form) with a heart in one lens — "kin", the people
// it helps you recognise — and a spark of memory in the other. Keep in sync with public/favicon.svg.
export function KinlyMark({ size = 30 }) {
  return (
    <svg
      className="kinly-mark"
      width={size * (44 / 26)}
      height={size}
      viewBox="0 0 44 26"
      aria-hidden="true"
    >
      <circle cx="11" cy="13" r="9" fill="rgba(255, 255, 255, 0.16)" stroke="currentColor" strokeWidth="2.6" />
      <circle cx="33" cy="13" r="9" fill="rgba(255, 255, 255, 0.16)" stroke="currentColor" strokeWidth="2.6" />
      <path d="M20 12.4 Q22 9.6 24 12.4" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
      <path
        className="kinly-accent"
        d="M11 17.4 C7.1 14.8 6.5 12.5 7.5 10.9 C8.5 9.4 10.4 9.5 11 11 C11.6 9.5 13.5 9.4 14.5 10.9 C15.5 12.5 14.9 14.8 11 17.4 Z"
      />
      <path className="kinly-accent" d="M33 8.4 L34.4 11.6 L37.6 13 L34.4 14.4 L33 17.6 L31.6 14.4 L28.4 13 L31.6 11.6 Z" />
    </svg>
  );
}

export default function KinlyLogo() {
  return (
    <div className="app-title" aria-label="Kinly">
      <KinlyMark />
      <span className="kinly-wordmark">Kinly</span>
    </div>
  );
}
