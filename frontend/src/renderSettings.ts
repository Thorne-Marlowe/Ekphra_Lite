export type SeedPolicy = 'random' | 'fixed'

const MAX_RANDOM_NOISE_SEED = 18_446_744_073_709_551_615n

/** Build the API payload without losing precision for unsigned 64-bit seeds. */
export function buildSeedSettings(policy: SeedPolicy, fixedSeed: string): { seedMode: SeedPolicy; seed?: string } {
  if (policy === 'random') return { seedMode: 'random' }

  const seed = fixedSeed.trim()
  if (!/^\d+$/.test(seed)) throw new Error('Fixed seed must be a whole number from 0 to 18446744073709551615.')
  if (BigInt(seed) > MAX_RANDOM_NOISE_SEED) {
    throw new Error('Fixed seed must be a whole number from 0 to 18446744073709551615.')
  }
  return { seedMode: 'fixed', seed }
}
