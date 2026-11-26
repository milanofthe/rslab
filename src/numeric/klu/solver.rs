//! The factored solver: the block substitutions and the factor exports.

use super::*;

impl<T: Scalar> crate::numeric::direct::SolveCore<T> for KluSolver<T> {
    const NAME: &'static str = "klu";

    fn dim(&self) -> usize {
        self.factors.n
    }

    fn counter(&self) -> &crate::diagnostics::SolveCounter {
        &self.solves
    }
    fn works(&self) -> &crate::numeric::direct::WorkPool<T> {
        &self.works
    }

    fn perturbations(&self) -> &[crate::PivotFloor<T>] {
        &[]
    }

    fn log_det_raw(&self) -> Result<crate::numeric::direct::LogDet<T>, RslabError> {
        let f = &self.factors;
        let mut d = crate::numeric::direct::LogDet::one();
        for &u in &f.udiag {
            d.mul(u);
        }
        d.permutation(&f.row_perm);
        d.permutation(&f.col_perm);
        for &s in &f.rs_inv {
            d.div_scale(s);
        }
        Ok(d)
    }

    fn solve_raw_into(
        &self,
        b: &[T],
        nrhs: usize,
        transpose: bool,
        x: &mut [T],
        work: &mut crate::SolveWork<T>,
    ) -> Result<(), RslabError> {
        let n = self.factors.n;
        if transpose {
            for c in 0..nrhs {
                let cols = c * n..(c + 1) * n;
                self.solve_transpose_one(&b[cols.clone()], &mut x[cols], &mut work.y);
            }
        } else if nrhs == 1 {
            self.solve_one(b, x, work);
        } else {
            self.solve_block(b, nrhs, x, work);
        }
        Ok(())
    }
}

crate::numeric::direct::direct_solver!(KluSolver);

impl<T: Scalar> KluSolver<T> {
    /// Heap bytes this factor holds: `L`, `U` and the off-block entries of
    /// every block, the permutations and scalings, and the refactorization's
    /// scatter program.
    pub fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        let f = &self.factors;
        vec_bytes(&f.block_ptr)
            + vec_bytes(&f.row_perm)
            + vec_bytes(&f.pinv)
            + vec_bytes(&f.col_perm)
            + vec_bytes(&f.rs_inv)
            + vec_bytes(&f.l_colptr)
            + vec_bytes(&f.l_rowidx)
            + vec_bytes(&f.l_val)
            + vec_bytes(&f.u_colptr)
            + vec_bytes(&f.u_rowidx)
            + vec_bytes(&f.u_val)
            + vec_bytes(&f.udiag)
            + vec_bytes(&f.sn_last)
            + vec_bytes(&f.u_dense)
            + vec_bytes(&f.f_colptr)
            + vec_bytes(&f.f_rowidx)
            + vec_bytes(&f.f_val)
            + vec_bytes(&f.scatter_expect)
            + vec_bytes(&f.scatter_target)
            + vec_bytes(&f.plan.blocks)
            + (f.plan.blocks.iter())
                .map(|p| vec_bytes(&p.level_ptr) + vec_bytes(&p.level_cols) + vec_bytes(&p.level_work))
                .sum::<u64>()
            + vec_bytes(&f.ready.done)
            + vec_bytes(&f.modes)
    }

    /// One-shot analyze + factor with the given settings. Skips the a-priori
    /// estimate and stage timing (empty [`diagnostics`](Self::diagnostics)),
    /// like [`LuSolver::factor`](crate::LuSolver::factor); use the phased
    /// [`KluSymbolic::factor`] for populated diagnostics.
    pub fn factor(a: &GeneralCsc<T>, settings: &KluSettings) -> Result<Self, RslabError> {
        KluSymbolic::analyze(a, settings)?.factor(a, settings)
    }

    /// Everything this factorization can tell about itself (see
    /// [`Diagnostics`](crate::Diagnostics)): the measured factor and refactor
    /// stages, the fill, the a-priori
    /// [`MemoryEstimate`](crate::diagnostics::MemoryEstimate) where
    /// [`KluSymbolic::estimate_memory`] computed it, and the solve-phase
    /// accumulators. A snapshot.
    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        let mut d = self.diagnostics.clone();
        d.solves = self.solves.snapshot();
        let u = &self.factors.udiag;
        d.numeric.min_pivot =
            (!u.is_empty()).then(|| u.iter().map(|v| v.magnitude()).fold(f64::MAX, f64::min));
        d
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Number of BTF diagonal blocks.
    pub fn n_blocks(&self) -> usize {
        self.factors.block_ptr.len() - 1
    }

    /// Stored factor entries: L + U + diagonal + off-block.
    pub fn factor_nnz(&self) -> usize {
        self.factors.l_val.len()
            + self.factors.u_val.len()
            + self.factors.udiag.len()
            + self.factors.f_val.len()
    }

    /// The unit lower factor `L`, diagonal included. With the row scaling
    /// `R = diag(row_scale)` and the permutations `P_r` (row `k` is row
    /// `row_perm[k]`) and `P_c` (column `k` is column `col_perm[k]`), the
    /// factorization is
    ///
    /// ```text
    /// P_r R A P_c = L U + F
    /// ```
    ///
    /// with `L` and `U` block diagonal over the blocks
    /// [`block_ptr`](Self::block_ptr) of the block triangular form and `F`
    /// the entries above the diagonal blocks.
    pub fn l_matrix(&self) -> GeneralCsc<T> {
        let f = &self.factors;
        factor_csc(f.n, &f.l_colptr, &f.l_rowidx, &f.l_val, |_| Some(T::one()))
    }

    /// The upper factor `U`, pivots on the diagonal (see
    /// [`l_matrix`](Self::l_matrix)).
    pub fn u_matrix(&self) -> GeneralCsc<T> {
        let f = &self.factors;
        factor_csc(f.n, &f.u_colptr, &f.u_rowidx, &f.u_val, |j| {
            Some(f.udiag[j])
        })
    }

    /// The entries `F` above the diagonal blocks (see
    /// [`l_matrix`](Self::l_matrix)).
    pub fn f_matrix(&self) -> GeneralCsc<T> {
        let f = &self.factors;
        factor_csc(f.n, &f.f_colptr, &f.f_rowidx, &f.f_val, |_| None)
    }

    /// The row permutation: row `k` of the factored matrix is row
    /// `row_perm[k]` of `A` (see [`l_matrix`](Self::l_matrix)).
    pub fn row_perm(&self) -> &[usize] {
        &self.factors.row_perm
    }

    /// The column permutation: column `k` of the factored matrix is column
    /// `col_perm[k]` of `A` (see [`l_matrix`](Self::l_matrix)).
    pub fn col_perm(&self) -> &[usize] {
        &self.factors.col_perm
    }

    /// Boundaries of the diagonal blocks of the block triangular form: block
    /// `b` holds the rows and columns `block_ptr[b]..block_ptr[b + 1]` of
    /// the factored matrix.
    pub fn block_ptr(&self) -> &[usize] {
        &self.factors.block_ptr
    }

    /// The row scaling: row `i` of `A` is multiplied by `row_scale[i]`
    /// before the factorization (all ones without scaling, see
    /// [`l_matrix`](Self::l_matrix)).
    pub fn row_scale(&self) -> &[f64] {
        &self.factors.rs_inv
    }

    fn solve_one(&self, b: &[T], x: &mut [T], work: &mut crate::SolveWork<T>) {
        let f = &self.factors;
        let w = &mut work.y;
        w.clear();
        w.extend(
            f.row_perm
                .iter()
                .map(|&orig| b[orig] * T::from_real(f.rs_inv[orig])),
        );
        debug_assert!(work.gather.iter().all(|v| *v == T::zero()));
        if work.gather.len() < f.l_widest {
            work.gather.resize(f.l_widest, T::zero());
        }
        self.solve_permuted(w, &mut work.gather);
        for (k, &c) in f.col_perm.iter().enumerate() {
            x[c] = w[k];
        }
    }

    fn solve_transpose_one(&self, b: &[T], x: &mut [T], w: &mut Vec<T>) {
        let f = &self.factors;
        w.clear();
        w.extend(f.col_perm.iter().map(|&c| b[c]));
        self.solve_permuted_transpose(w);
        for (k, &orig) in f.row_perm.iter().enumerate() {
            x[orig] = w[k] * T::from_real(f.rs_inv[orig]);
        }
    }

    fn solve_permuted_transpose(&self, w: &mut [T]) {
        let f = &self.factors;
        for b in 0..f.block_ptr.len() - 1 {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            for j in bs..be {
                let mut acc = w[j];
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    acc = acc - f.f_val[k] * w[f.f_rowidx[k] as usize];
                }
                w[j] = acc;
            }
            for j in bs..be {
                let mut acc = w[j];
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    acc = acc - f.u_val[k] * w[f.u_rowidx[k] as usize];
                }
                w[j] = acc / f.udiag[j];
            }
            for j in (bs..be).rev() {
                let mut acc = w[j];
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    acc = acc - f.l_val[k] * w[f.l_rowidx[k] as usize];
                }
                w[j] = acc;
            }
        }
    }

    fn solve_block(&self, b: &[T], nrhs: usize, x: &mut [T], work: &mut crate::SolveWork<T>) {
        let f = &self.factors;
        let n = f.n;
        work.y.clear();
        work.y.resize(n * nrhs, T::zero());
        let w = work.y.as_mut_slice();
        for (k, &orig) in f.row_perm.iter().enumerate() {
            let sv = T::from_real(f.rs_inv[orig]);
            for c in 0..nrhs {
                w[k * nrhs + c] = b[c * n + orig] * sv;
            }
        }
        work.row.clear();
        work.row.resize(nrhs, T::zero());
        let xj = work.row.as_mut_slice();
        for blk in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[blk], f.block_ptr[blk + 1]);
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    let (lr, nlv) = (f.l_rowidx[k] as usize, T::zero() - f.l_val[k]);
                    let row = &mut w[lr * nrhs..lr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&*xj) {
                        *r = fmadd(nlv, x, *r);
                    }
                }
            }
            for j in (bs..be).rev() {
                let d = f.udiag[j];
                {
                    let row = &mut w[j * nrhs..j * nrhs + nrhs];
                    for r in row.iter_mut() {
                        *r = *r / d;
                    }
                }
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    let (ur, nuv) = (f.u_rowidx[k] as usize, T::zero() - f.u_val[k]);
                    let row = &mut w[ur * nrhs..ur * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&*xj) {
                        *r = fmadd(nuv, x, *r);
                    }
                }
            }
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    let (fr, nfv) = (f.f_rowidx[k] as usize, T::zero() - f.f_val[k]);
                    let row = &mut w[fr * nrhs..fr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&*xj) {
                        *r = fmadd(nfv, x, *r);
                    }
                }
            }
        }
        for (k, &col) in f.col_perm.iter().enumerate() {
            for c in 0..nrhs {
                x[c * n + col] = w[k * nrhs + c];
            }
        }
    }

    fn solve_permuted(&self, w: &mut [T], tmp: &mut [T]) {
        let f = &self.factors;
        for b in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            let mut j = bs;
            while j < be {
                let q = f.sn_last[j] as usize;
                if q > j {
                    // SAFETY: the rows of `L(:, q)` are final positions in
                    // `w`, `tmp` holds `l_widest` entries.
                    unsafe {
                        super::supernode::apply(
                            w,
                            0,
                            tmp,
                            &f.l_colptr,
                            &f.l_rowidx,
                            (f.l_val.as_ptr(), 0),
                            (j, q),
                        );
                    }
                    j = q + 1;
                    continue;
                }
                let xj = w[j];
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                        let lr = f.l_rowidx[k] as usize;
                        debug_assert!(lr < w.len());
                        unsafe {
                            *w.get_unchecked_mut(lr) = fmadd(f.l_val[k], nxj, *w.get_unchecked(lr));
                        }
                    }
                }
                j += 1;
            }
            for j in (bs..be).rev() {
                let xj = w[j] / f.udiag[j];
                w[j] = xj;
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                        let ur = f.u_rowidx[k] as usize;
                        debug_assert!(ur < w.len());
                        unsafe {
                            *w.get_unchecked_mut(ur) = fmadd(f.u_val[k], nxj, *w.get_unchecked(ur));
                        }
                    }
                }
            }
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                        let fr = f.f_rowidx[k] as usize;
                        debug_assert!(fr < w.len());
                        unsafe {
                            *w.get_unchecked_mut(fr) = fmadd(f.f_val[k], nxj, *w.get_unchecked(fr));
                        }
                    }
                }
            }
        }
    }
}
