//! Deterministic linear congruential generator (Knuth's MMIX constants).
use num_complex::Complex;

/// `len` complex numbers whose real and imaginary parts are uniform in
/// `[-0.5, 0.5)`, from the generator seeded with `seed`.
pub fn complex(len: usize, seed: u64) -> Vec<Complex<f64>> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..len).map(|_| Complex::new(next(), next())).collect()
}
