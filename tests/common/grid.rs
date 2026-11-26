/// The `(rows, cols)` of an `m x m` grid Laplacian's pattern.
pub fn grid(m: usize) -> (Vec<usize>, Vec<usize>) {
    let (mut rows, mut cols) = (Vec::new(), Vec::new());
    for i in 0..m * m {
        rows.push(i);
        cols.push(i);
        for j in [i + 1, i + m] {
            if j < m * m && (j != i + 1 || j % m != 0) {
                rows.extend([i, j]);
                cols.extend([j, i]);
            }
        }
    }
    (rows, cols)
}
