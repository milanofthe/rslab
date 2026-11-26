//! The pattern-only left-looking schedule: per-supernode row structures and
//! updater lists, and the consumer refcounts that free a panel after its
//! last updater.

pub(crate) fn emit_refcount_offsets(
    sym: &crate::symbolic::SymbolicFactorization,
    sched: &LlSchedule,
) -> (Vec<std::sync::atomic::AtomicUsize>, Vec<usize>) {
    use std::sync::atomic::AtomicUsize;
    let nsuper = sym.supernodes.len();
    let mut refcount: Vec<AtomicUsize> = (0..nsuper).map(|_| AtomicUsize::new(0)).collect();
    for s in 0..nsuper {
        for &k in sched.updaters(s) {
            *refcount[k as usize].get_mut() += 1;
        }
    }
    let mut e_offset = vec![0usize; nsuper];
    let mut acc = 0usize;
    for (s, snode) in sym.supernodes.iter().enumerate() {
        e_offset[s] = acc;
        acc += snode.ncol;
    }
    (refcount, e_offset)
}

pub(crate) type Li = u32;

pub(crate) struct LlSchedule {
    rs_off: Vec<usize>,
    rs: Vec<Li>,
    ul_off: Vec<usize>,
    ul: Vec<Li>,
    pub(crate) parent: Vec<usize>,
    pub(crate) leaves: Vec<usize>,
}

impl LlSchedule {
    pub(crate) fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        vec_bytes(&self.rs_off)
            + vec_bytes(&self.rs)
            + vec_bytes(&self.ul_off)
            + vec_bytes(&self.ul)
            + vec_bytes(&self.parent)
            + vec_bytes(&self.leaves)
    }

    /// Rows of supernode `s`: `rows(s)[0..ncol]` are its eliminated columns
    /// `first_col..first_col+ncol`; `rows(s)[ncol..]` the sorted
    /// below-diagonal rows of the panel.
    #[inline]
    pub fn rows(&self, s: usize) -> &[Li] {
        &self.rs[self.rs_off[s]..self.rs_off[s + 1]]
    }

    /// Updaters of supernode `s`: every factored `k` whose off-diagonal rows
    /// hit `s`'s column run (each exactly once, ascending).
    #[inline]
    pub fn updaters(&self, s: usize) -> &[Li] {
        &self.ul[self.ul_off[s]..self.ul_off[s + 1]]
    }

    pub fn build(sym: &crate::symbolic::SymbolicFactorization) -> Self {
        let nsuper = sym.supernodes.len();
        assert!(
            sym.n <= Li::MAX as usize,
            "left-looking schedule requires n < 2^32"
        );
        let mut rs_off = Vec::with_capacity(nsuper + 1);
        rs_off.push(0usize);
        let mut rs: Vec<Li> = Vec::new();
        let mut trailing: Vec<Li> = Vec::new();
        let steps = crate::progress::Steps::new(rsprobe::active!(), sym.n as u64);
        for s in 0..nsuper {
            let snode = &sym.supernodes[s];
            if let Some(step) = steps.as_ref().and_then(|st| st.add(snode.ncol as u64)) {
                rsprobe::progress!(step, crate::constants::PROGRESS_STEPS; "supernode structure");
            }
            let own_last = snode.first_col + snode.ncol;
            trailing.clear();
            for j in snode.first_col..own_last {
                for &r in sym.pattern.col(sym.perm[j]) {
                    let r = sym.perm_inv[r as usize];
                    if r >= own_last {
                        trailing.push(r as Li);
                    }
                }
            }
            for &ch in &snode.children {
                let nck = sym.supernodes[ch].ncol;
                for &r in &rs[rs_off[ch] + nck..rs_off[ch + 1]] {
                    if r as usize >= own_last {
                        trailing.push(r);
                    }
                }
            }
            trailing.sort_unstable();
            trailing.dedup();
            rs.extend((snode.first_col..own_last).map(|c| c as Li));
            rs.extend_from_slice(&trailing);
            rs_off.push(rs.len());
        }

        let mut col_to_snode = vec![0usize; sym.n];
        for (s, snode) in sym.supernodes.iter().enumerate() {
            col_to_snode[snode.first_col..snode.first_col + snode.ncol].fill(s);
        }
        let mut ul_off = vec![0usize; nsuper + 1];
        let each_hit = |mut f: Box<dyn FnMut(usize, usize) + '_>| {
            for k in 0..nsuper {
                let nck = sym.supernodes[k].ncol;
                let mut last = usize::MAX;
                for &r in &rs[rs_off[k] + nck..rs_off[k + 1]] {
                    let s = col_to_snode[r as usize];
                    if s != last {
                        f(s, k);
                        last = s;
                    }
                }
            }
        };
        each_hit(Box::new(|s, _k| ul_off[s + 1] += 1));
        for s in 0..nsuper {
            ul_off[s + 1] += ul_off[s];
        }
        let mut cursor = ul_off[..nsuper].to_vec();
        let mut ul = vec![0 as Li; ul_off[nsuper]];
        each_hit(Box::new(|s, k| {
            ul[cursor[s]] = k as Li;
            cursor[s] += 1;
        }));
        let (parent, leaves) = forest(sym);
        LlSchedule {
            rs_off,
            rs,
            ul_off,
            ul,
            parent,
            leaves,
        }
    }
}

fn forest(sym: &crate::symbolic::SymbolicFactorization) -> (Vec<usize>, Vec<usize>) {
    let nodes = &sym.supernodes;
    let mut is_child = vec![false; nodes.len()];
    for sn in nodes {
        for &c in &sn.children {
            is_child[c] = true;
        }
    }
    let mut parent = vec![usize::MAX; nodes.len()];
    let mut leaves = Vec::new();
    let mut stack: Vec<usize> = (0..nodes.len()).filter(|&s| !is_child[s]).collect();
    while let Some(s) = stack.pop() {
        if nodes[s].children.is_empty() {
            leaves.push(s);
        }
        for &c in &nodes[s].children {
            debug_assert!(c < s, "children come before their parent");
            parent[c] = s;
            stack.push(c);
        }
    }
    (parent, leaves)
}
