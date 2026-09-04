/** Presentation-only formatting helpers. No business logic. */

export function formatMoney(amount: number, currency: string): string {
  return `${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${currency}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}

export function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}
