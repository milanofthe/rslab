use std::collections::BTreeSet;

pub fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
    let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(i, j) in triples {
        set.insert((i, j));
        set.insert((j, i));
    }
    let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
    for &(r, c) in &set {
        cols[c].push(r as i32);
    }
    for col in &mut cols {
        col.sort();
    }
    let mut col_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx: Vec<i32> = Vec::new();
    for col in &cols {
        for &r in col {
            row_idx.push(r);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx)
}

pub fn arrow(n: usize) -> (Vec<i32>, Vec<i32>) {
    let mut t = Vec::new();
    for i in 0..n {
        t.push((i, i));
    }
    for i in 1..n {
        t.push((0, i));
    }
    csc_from_triples(n, &t)
}

pub fn grid_2d(m: usize, n: usize) -> (Vec<i32>, Vec<i32>) {
    let idx = |r: usize, c: usize| r * n + c;
    let total = m * n;
    let mut t = Vec::new();
    for r in 0..m {
        for c in 0..n {
            let k = idx(r, c);
            t.push((k, k));
            if r + 1 < m {
                t.push((k, idx(r + 1, c)));
            }
            if c + 1 < n {
                t.push((k, idx(r, c + 1)));
            }
        }
    }
    csc_from_triples(total, &t)
}
