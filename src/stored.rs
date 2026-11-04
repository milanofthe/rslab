//! Values stored in a field's single-precision twin where their accuracy allows,
//! read and computed with in the field itself.

use crate::constants::SINGLE_MARGIN;
use crate::scalar::Scalar;

/// Whether data accurate to the relative tolerance `tol` is stored in the
/// single-precision twin of `T` ([`Stored`]): where
/// [`SINGLE_MARGIN`](crate::constants::SINGLE_MARGIN) times the twin's unit
/// roundoff is at most `tol`, so the rounding adds at most a sixteenth of `tol`
/// to the error the data holds (`Complex<f64>`: from `tol = 16 * 6e-8 = 9.5e-7` on).
pub fn single_for<T: Scalar>(tol: f64) -> bool {
    tol >= SINGLE_MARGIN * <T::Low as Scalar>::EPSILON / 2.0
}

/// A buffer of values of the field `T`, stored in `T` or in its single-precision
/// twin [`T::Low`](Scalar::Low) (half the memory for the double fields): written
/// from `T` (rounded to the storage), read as `T` (widened, exactly), and its
/// products ([`dotc`](Self::dotc), [`axpy`](Self::axpy)) run in `T`, each stored
/// value widened as it is loaded, so no copy in `T` is formed. Whether to
/// store single: [`single_for`] of the tolerance the values are accurate to.
///
/// A block GMRES keeps its Krylov bases in it
/// ([`KrylovSettings::basis_single`](crate::KrylovSettings::basis_single)).
///
/// ```
/// use num_complex::Complex64;
/// use rslab::Stored;
/// let x = [Complex64::new(1.0 / 3.0, -2.0); 4];
/// let s = Stored::new(&x, true);
/// assert_eq!(s.bytes(), 4 * 8);
/// assert!((s.at(2) - x[2]).norm() <= f32::EPSILON as f64);
/// ```
#[derive(Clone, Debug)]
pub struct Stored<T: Scalar> {
    repr: Repr<T>,
}

#[derive(Clone, Debug)]
enum Repr<T: Scalar> {
    Full(Vec<T>),
    Single(Vec<T::Low>),
}

impl<T: Scalar> Default for Stored<T> {
    /// Empty, in `T`.
    fn default() -> Self {
        Self::with_capacity(0, false)
    }
}

impl<T: Scalar> Stored<T> {
    /// The values `x`, in the twin with `single`.
    pub fn new(x: &[T], single: bool) -> Self {
        let mut s = Self::with_capacity(x.len(), single);
        s.write(0, x);
        s
    }

    /// Empty, `cap` values reserved (address space only: what is written is resident),
    /// in the twin with `single`.
    pub fn with_capacity(cap: usize, single: bool) -> Self {
        let repr = if single {
            Repr::Single(Vec::with_capacity(cap))
        } else {
            Repr::Full(Vec::with_capacity(cap))
        };
        Stored { repr }
    }

    /// Bytes a value takes stored, in the twin with `single`.
    pub fn scalar_bytes(single: bool) -> usize {
        if single {
            std::mem::size_of::<T::Low>()
        } else {
            std::mem::size_of::<T>()
        }
    }

    /// Whether it stores in the twin.
    pub fn is_single(&self) -> bool {
        matches!(self.repr, Repr::Single(_))
    }

    /// The values written.
    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Full(v) => v.len(),
            Repr::Single(v) => v.len(),
        }
    }

    /// Whether no value is written.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values reserved.
    pub fn capacity(&self) -> usize {
        match &self.repr {
            Repr::Full(v) => v.capacity(),
            Repr::Single(v) => v.capacity(),
        }
    }

    /// Heap bytes of the values written.
    pub fn bytes(&self) -> u64 {
        (self.len() * Self::scalar_bytes(self.is_single())) as u64
    }

    /// Value `i`.
    #[inline]
    pub fn at(&self, i: usize) -> T {
        match &self.repr {
            Repr::Full(v) => v[i],
            Repr::Single(v) => T::promote(v[i]),
        }
    }

    /// `out[i] = self[from + i]`.
    pub fn read(&self, from: usize, out: &mut [T]) {
        let to = from + out.len();
        match &self.repr {
            Repr::Full(v) => out.copy_from_slice(&v[from..to]),
            Repr::Single(v) => out
                .iter_mut()
                .zip(&v[from..to])
                .for_each(|(o, &a)| *o = T::promote(a)),
        }
    }

    /// `self[from + i] = x[i]`, rounded to the storage; it grows (zero filled) where it
    /// ends before `from + x.len()`, inside the capacity without moving.
    pub fn write(&mut self, from: usize, x: &[T]) {
        let to = from + x.len();
        match &mut self.repr {
            Repr::Full(v) => {
                if v.len() < to {
                    v.resize(to, T::zero());
                }
                v[from..to].copy_from_slice(x);
            }
            Repr::Single(v) => {
                if v.len() < to {
                    v.resize(to, T::Low::zero());
                }
                v[from..to]
                    .iter_mut()
                    .zip(x)
                    .for_each(|(o, &a)| *o = a.demote());
            }
        }
    }

    /// The conjugated inner product `sum conj(self[from + i]) * w[i]`, in order.
    #[inline]
    pub fn dotc(&self, from: usize, w: &[T]) -> T {
        let to = from + w.len();
        match &self.repr {
            Repr::Full(v) => dotc_as(&v[from..to], w, |a| a),
            Repr::Single(v) => dotc_as(&v[from..to], w, T::promote),
        }
    }

    /// `out[i] += a * self[from + i]`.
    #[inline]
    pub fn axpy(&self, from: usize, a: T, out: &mut [T]) {
        let to = from + out.len();
        match &self.repr {
            Repr::Full(v) => axpy_as(&v[from..to], a, out, |x| x),
            Repr::Single(v) => axpy_as(&v[from..to], a, out, T::promote),
        }
    }

    /// Store the values in the twin from now on: those written are rounded, the
    /// capacity is kept. They are copied, so while it runs the values are held twice.
    pub fn round_to_single(&mut self) {
        if let Repr::Full(v) = &self.repr {
            let mut low = Vec::with_capacity(v.capacity());
            low.extend(v.iter().map(|a| a.demote()));
            self.repr = Repr::Single(low);
        }
    }

    /// The stored values' little-endian bytes, in the precision stored (a fingerprint).
    pub fn le_bytes(&self, out: &mut Vec<u8>) {
        match &self.repr {
            Repr::Full(v) => v.iter().for_each(|a| a.le_bytes(out)),
            Repr::Single(v) => v.iter().for_each(|a| a.le_bytes(out)),
        }
    }
}

/// `sum conj(x[i]) * y[i]` in order, `x` stored in `E` and widened by `load`.
#[inline]
pub(crate) fn dotc_as<T: Scalar, E: Copy>(x: &[E], y: &[T], load: impl Fn(E) -> T) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + load(xi).conj() * yi;
    }
    s
}

/// `out[i] += a * x[i]`, `x` stored in `E` and widened by `load`.
#[inline]
pub(crate) fn axpy_as<T: Scalar, E: Copy>(x: &[E], a: T, out: &mut [T], load: impl Fn(E) -> T) {
    for (o, &xi) in out.iter_mut().zip(x) {
        *o = *o + a * load(xi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    type C = Complex<f64>;

    /// Stored single, a value is off by at most the twin's unit roundoff relative to
    /// its components, read and products widen exactly, the bytes halve; stored in
    /// `T`, everything is bit for bit.
    #[test]
    fn stored_rounds_once_and_widens_exactly() {
        let x: Vec<C> = (0..37)
            .map(|i| C::new((i as f64).sin() / 3.0, (i as f64 * 0.7).cos() * 1e3))
            .collect();
        let (full, single) = (Stored::new(&x, false), Stored::new(&x, true));
        assert_eq!(full.bytes(), 2 * single.bytes());
        let u = f32::EPSILON as f64 / 2.0;
        for (i, &a) in x.iter().enumerate() {
            assert_eq!(full.at(i), a);
            let e = single.at(i) - a;
            assert!(e.re.abs() <= u * a.re.abs() && e.im.abs() <= u * a.im.abs());
        }
        let mut wide = vec![C::default(); 30];
        single.read(5, &mut wide);
        let w: Vec<C> = (0..30).map(|i| C::new(1.0, i as f64)).collect();
        let dot = (0..30).fold(C::default(), |s, i| s + wide[i].conj() * w[i]);
        assert_eq!(single.dotc(5, &w), dot);
        assert_eq!(full.dotc(5, &w), dotc_as(&x[5..35], &w, |a| a));
        let mut out = w.clone();
        single.axpy(5, C::new(0.5, -2.0), &mut out);
        for i in 0..30 {
            assert_eq!(out[i], w[i] + C::new(0.5, -2.0) * wide[i]);
        }
        let (mut fb, mut sb) = (Vec::new(), Vec::new());
        full.le_bytes(&mut fb);
        single.le_bytes(&mut sb);
        assert_eq!((fb.len(), sb.len()), (37 * 16, 37 * 8));
    }

    /// Rounded to the twin later, a store holds what one written single holds, its
    /// capacity kept; one single already stays as it is.
    #[test]
    fn stored_rounds_to_single_in_place() {
        let x: Vec<C> = (0..19)
            .map(|i| C::new((i as f64).cos() / 7.0, i as f64 * 0.3))
            .collect();
        let mut s = Stored::<C>::with_capacity(32, false);
        s.write(0, &x);
        s.round_to_single();
        let direct = Stored::new(&x, true);
        assert!(s.is_single() && s.capacity() >= 32);
        assert_eq!((s.len(), s.bytes()), (direct.len(), direct.bytes()));
        let (mut a, mut b) = (Vec::new(), Vec::new());
        s.le_bytes(&mut a);
        direct.le_bytes(&mut b);
        assert_eq!(a, b);
        s.round_to_single();
        let mut c = Vec::new();
        s.le_bytes(&mut c);
        assert_eq!(a, c);
    }

    /// A write past the end grows the store, zero filled, within the capacity.
    #[test]
    fn stored_write_grows_zero_filled() {
        let mut s = Stored::<C>::with_capacity(8, true);
        assert!(s.is_empty() && s.capacity() >= 8);
        s.write(4, &[C::new(1.0, 2.0); 2]);
        assert_eq!(s.len(), 6);
        assert_eq!(s.at(3), C::default());
        assert_eq!(s.at(5), C::new(1.0, 2.0));
    }

    /// The precision rule: single from sixteen units of the twin's roundoff on.
    #[test]
    fn single_from_sixteen_roundoffs_on() {
        let edge = SINGLE_MARGIN * f32::EPSILON as f64 / 2.0;
        assert!(single_for::<C>(edge) && single_for::<C>(1e-6));
        assert!(!single_for::<C>(edge * 0.99) && !single_for::<f64>(1e-8));
    }
}
