#[cfg(test)]
use crate::dense::matrix::SymmetricMatrix;
use crate::error::RslabError;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;
use std::path::Path;

/// A sparse symmetric matrix in coordinate (COO) format, as read from a Matrix
/// Market file. Entries are 0-indexed, lower triangle only (i >= j). Generic
/// over the scalar field: [`f64`] (`real symmetric`, PARDISO `mtype 2`) or
/// [`Complex<f64>`] (`complex symmetric`, PARDISO `mtype 6`).
#[derive(Debug, Clone)]
pub struct MtxMatrix<T = f64> {
    pub n: usize,
    pub entries: Vec<(usize, usize, T)>,
}

impl<T: Scalar> MtxMatrix<T> {
    #[cfg(test)]
    pub(crate) fn to_dense(&self) -> SymmetricMatrix<T> {
        let mut mat = SymmetricMatrix::zeros(self.n);
        for &(i, j, v) in &self.entries {
            let prev = mat.get(i, j);
            mat.set(i, j, prev + v);
        }
        mat
    }

    /// Convert to a CSC sparse matrix (lower triangle).
    pub fn to_csc(&self) -> Result<CscMatrix<T>, RslabError> {
        let rows: Vec<usize> = self.entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = self.entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<T> = self.entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(self.n, &rows, &cols, &vals)
    }

    /// Convert to a general (full, unsymmetric) CSC matrix, keeping every entry
    /// as given - the form the unsymmetric LU path / a `LinearOperator` uses.
    pub fn to_general_csc(&self) -> Result<GeneralCsc<T>, RslabError> {
        let rows: Vec<usize> = self.entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = self.entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<T> = self.entries.iter().map(|&(_, _, v)| v).collect();
        GeneralCsc::from_triplets(self.n, &rows, &cols, &vals)
    }
}

/// Read a **real** symmetric Matrix Market file (`mtype 2`).
///
/// Accepts only `%%MatrixMarket matrix coordinate real symmetric`. Indices are
/// converted from 1-based (MTX) to 0-based; upper-triangle entries are
/// transposed to the lower triangle.
pub fn read_mtx(path: &Path) -> Result<MtxMatrix<f64>, RslabError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| RslabError::IoError(format!("{}: {}", path.display(), e)))?;
    parse_mtx(&contents, path.to_string_lossy().as_ref())
}

/// Read a **complex** symmetric Matrix Market file (`mtype 6`).
///
/// Accepts `%%MatrixMarket matrix coordinate complex symmetric`. Each data line
/// is `i j re im`. Indices 1-based -> 0-based; upper triangle -> lower triangle.
pub fn read_mtx_complex(path: &Path) -> Result<MtxMatrix<Complex<f64>>, RslabError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| RslabError::IoError(format!("{}: {}", path.display(), e)))?;
    parse_mtx_complex(&contents, path.to_string_lossy().as_ref())
}

/// A Matrix Market file loaded with its structure tag, values widened to
/// `Complex<f64>` (a real input gets a zero imaginary part). Auto-detected by
/// [`read_mtx_any`] so a heterogeneous corpus (SuiteSparse) routes each matrix to
/// the right solver path.
pub enum MtxLoaded {
    /// `symmetric` file: lower triangle, for the LDL^T path.
    Symmetric(CscMatrix<Complex<f64>>),
    /// `general` file: full pattern, for the LU path.
    General(GeneralCsc<Complex<f64>>),
}

/// Read any `real`/`complex`/`integer` x `symmetric`/`general` Matrix Market
/// coordinate file, widening values to `Complex<f64>`. The header is auto-detected;
/// `pattern`, `hermitian`, and `skew-symmetric` are rejected (not the
/// complex-symmetric / general forms the solvers take).
pub fn read_mtx_any(path: &Path) -> Result<MtxLoaded, RslabError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| RslabError::IoError(format!("{}: {}", path.display(), e)))?;
    let source = path.to_string_lossy();
    let header = contents.lines().next().unwrap_or("");
    let t: Vec<String> = header
        .split_whitespace()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    if t.len() != crate::constants::MTX_BANNER_WORDS
        || t[0] != "%%matrixmarket"
        || t[1] != "matrix"
        || t[2] != "coordinate"
    {
        return Err(RslabError::IoError(format!(
            "{source}: unsupported header '{}'",
            header.trim()
        )));
    }
    let (field, symmetry) = (t[3].clone(), t[4].clone());
    if field != "real" && field != "complex" && field != "integer" {
        return Err(RslabError::IoError(format!(
            "{source}: unsupported field '{field}' (real/complex/integer only)"
        )));
    }
    let is_complex = field == "complex";
    let nvt = if is_complex { 2 } else { 1 };
    let to_c = move |toks: &[&str]| -> Option<Complex<f64>> {
        let re = toks.first()?.parse::<f64>().ok()?;
        let im = if is_complex {
            toks.get(1)?.parse::<f64>().ok()?
        } else {
            0.0
        };
        Some(Complex::new(re, im))
    };
    match symmetry.as_str() {
        "symmetric" => {
            let m = parse_mtx_with(&contents, &source, &field, "symmetric", nvt, to_c)?;
            Ok(MtxLoaded::Symmetric(m.to_csc()?))
        }
        "general" => {
            let m = parse_mtx_with(&contents, &source, &field, "general", nvt, to_c)?;
            Ok(MtxLoaded::General(m.to_general_csc()?))
        }
        other => Err(RslabError::IoError(format!(
            "{source}: unsupported symmetry '{other}' (symmetric/general only)"
        ))),
    }
}

/// Parse a **real** symmetric Matrix Market string (`mtype 2`).
pub fn parse_mtx(contents: &str, source: &str) -> Result<MtxMatrix<f64>, RslabError> {
    parse_mtx_with(contents, source, "real", "symmetric", 1, |toks| {
        toks[0].parse::<f64>().ok()
    })
}

/// Parse a **complex** symmetric Matrix Market string (`mtype 6`). Data lines
/// carry two value tokens, the real and imaginary parts: `i j re im`.
pub fn parse_mtx_complex(
    contents: &str,
    source: &str,
) -> Result<MtxMatrix<Complex<f64>>, RslabError> {
    parse_mtx_with(contents, source, "complex", "symmetric", 2, |toks| {
        let re = toks[0].parse::<f64>().ok()?;
        let im = toks[1].parse::<f64>().ok()?;
        Some(Complex::new(re, im))
    })
}

/// Parse a **complex general** Matrix Market string (both triangles stored, e.g.
/// MoM near-field matrices written as `general`). Entries are kept as given
/// (0-indexed, no lower-triangle folding); the caller decides how to treat
/// symmetry (e.g. extract the lower triangle for the symmetric solver, or feed
/// the full pattern to a general LU). Data lines are `i j re im`.
pub fn parse_mtx_complex_general(
    contents: &str,
    source: &str,
) -> Result<MtxMatrix<Complex<f64>>, RslabError> {
    parse_mtx_with(contents, source, "complex", "general", 2, |toks| {
        let re = toks[0].parse::<f64>().ok()?;
        let im = toks[1].parse::<f64>().ok()?;
        Some(Complex::new(re, im))
    })
}

/// `a` as a Matrix Market `coordinate general` file: `real` or `complex` by
/// the scalar field, 1-based indices, values in round-trip exponent form
/// (the format [`read_mtx_any`] reads back).
pub fn format_mtx<T: Scalar>(a: &GeneralCsc<T>) -> String {
    format_coordinate(a.n, &a.col_ptr, &a.row_idx, &a.values, "general")
}

/// The lower triangle `a` as a Matrix Market `coordinate symmetric` file (see
/// [`format_mtx`]).
pub fn format_mtx_symmetric<T: Scalar>(a: &CscMatrix<T>) -> String {
    format_coordinate(a.n, &a.col_ptr, &a.row_idx, &a.values, "symmetric")
}

/// The vector `v` as a Matrix Market `array general` file with one column
/// (a right-hand side), values as in [`format_mtx`].
pub fn format_mtx_array<T: Scalar>(v: &[T]) -> String {
    let mut out = format!(
        "%%MatrixMarket matrix array {} general\n{} 1\n",
        field_of::<T>(),
        v.len()
    );
    for &x in v {
        push_value(&mut out, x);
        out.push('\n');
    }
    out
}

/// Write [`format_mtx`] of `a` to `path`.
pub fn write_mtx<T: Scalar>(path: &Path, a: &GeneralCsc<T>) -> Result<(), RslabError> {
    write_text(path, format_mtx(a))
}

/// Write [`format_mtx_symmetric`] of `a` to `path`.
pub fn write_mtx_symmetric<T: Scalar>(path: &Path, a: &CscMatrix<T>) -> Result<(), RslabError> {
    write_text(path, format_mtx_symmetric(a))
}

/// Write [`format_mtx_array`] of `v` to `path`.
pub fn write_mtx_array<T: Scalar>(path: &Path, v: &[T]) -> Result<(), RslabError> {
    write_text(path, format_mtx_array(v))
}

fn write_text(path: &Path, text: String) -> Result<(), RslabError> {
    std::fs::write(path, text)
        .map_err(|e| RslabError::IoError(format!("{}: {}", path.display(), e)))
}

fn field_of<T: Scalar>() -> &'static str {
    if T::COMPLEX {
        "complex"
    } else {
        "real"
    }
}

fn push_value<T: Scalar>(out: &mut String, x: T) {
    use std::fmt::Write;
    let _ = if T::COMPLEX {
        write!(out, "{:e} {:e}", x.real(), x.imag())
    } else {
        write!(out, "{:e}", x.real())
    };
}

fn format_coordinate<T: Scalar>(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[T],
    symmetry: &str,
) -> String {
    use std::fmt::Write;
    let mut out = format!(
        "%%MatrixMarket matrix coordinate {} {symmetry}\n{n} {n} {}\n",
        field_of::<T>(),
        row_idx.len()
    );
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let _ = write!(out, "{} {} ", row_idx[p] + 1, j + 1);
            push_value(&mut out, values[p]);
            out.push('\n');
        }
    }
    out
}

fn parse_mtx_with<T: Scalar>(
    contents: &str,
    source: &str,
    field: &str,
    symmetry: &str,
    n_value_tokens: usize,
    parse_val: impl Fn(&[&str]) -> Option<T>,
) -> Result<MtxMatrix<T>, RslabError> {
    let fold_to_lower = symmetry == "symmetric";
    let mut lines = contents.lines().enumerate();

    let (_, header) = lines
        .next()
        .ok_or_else(|| RslabError::IoError(format!("{}: empty file", source)))?;
    // https://math.nist.gov/MatrixMarket/formats.html
    let banner: [&str; 5] = ["%%matrixmarket", "matrix", "coordinate", field, symmetry];
    let mut header_tokens = header.split_whitespace();
    let banner_ok = banner.iter().all(|expected| {
        header_tokens
            .next()
            .is_some_and(|tok| tok.eq_ignore_ascii_case(expected))
    }) && header_tokens.next().is_none();
    if !banner_ok {
        return Err(RslabError::IoError(format!(
            "{}: unsupported header '{}' (expected: %%MatrixMarket matrix coordinate {} {})",
            source,
            header.trim(),
            field,
            symmetry
        )));
    }

    let mut size_line: Option<(usize, String)> = None;
    for (line_no, line) in &mut lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        size_line = Some((line_no, trimmed.to_string()));
        break;
    }

    let (size_line_no, size_text) =
        size_line.ok_or_else(|| RslabError::IoError(format!("{}: missing size line", source)))?;

    let parts: Vec<&str> = size_text.split_whitespace().collect();
    if parts.len() != crate::constants::MTX_SIZE_FIELDS {
        return Err(RslabError::IoError(format!(
            "{}: line {}: expected 'rows cols nnz', got '{}'",
            source,
            size_line_no + 1,
            size_text
        )));
    }
    let rows: usize = parts[0].parse().map_err(|_| {
        RslabError::IoError(format!(
            "{}: line {}: invalid row count '{}'",
            source,
            size_line_no + 1,
            parts[0]
        ))
    })?;
    let cols: usize = parts[1].parse().map_err(|_| {
        RslabError::IoError(format!(
            "{}: line {}: invalid col count '{}'",
            source,
            size_line_no + 1,
            parts[1]
        ))
    })?;
    let nnz: usize = parts[2].parse().map_err(|_| {
        RslabError::IoError(format!(
            "{}: line {}: invalid nnz '{}'",
            source,
            size_line_no + 1,
            parts[2]
        ))
    })?;

    if rows != cols {
        return Err(RslabError::IoError(format!(
            "{}: symmetric matrix must be square, got {}x{}",
            source, rows, cols
        )));
    }
    let n = rows;

    let mut entries = Vec::with_capacity(nnz.min(contents.len()));
    for (line_no, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let expected_tokens = 2 + n_value_tokens;
        if parts.len() != expected_tokens {
            return Err(RslabError::IoError(format!(
                "{}: line {}: expected 'i j {}', got '{}'",
                source,
                line_no + 1,
                if n_value_tokens == 2 {
                    "re im"
                } else {
                    "value"
                },
                trimmed
            )));
        }
        let i: usize = parts[0].parse().map_err(|_| {
            RslabError::IoError(format!(
                "{}: line {}: invalid row index '{}'",
                source,
                line_no + 1,
                parts[0]
            ))
        })?;
        let j: usize = parts[1].parse().map_err(|_| {
            RslabError::IoError(format!(
                "{}: line {}: invalid col index '{}'",
                source,
                line_no + 1,
                parts[1]
            ))
        })?;
        let v: T = parse_val(&parts[2..]).ok_or_else(|| {
            RslabError::IoError(format!(
                "{}: line {}: invalid value '{}'",
                source,
                line_no + 1,
                parts[2..].join(" ")
            ))
        })?;
        if !v.is_finite() {
            return Err(RslabError::IoError(format!(
                "{}: line {}: non-finite value '{}'",
                source,
                line_no + 1,
                parts[2..].join(" ")
            )));
        }

        if i == 0 || j == 0 || i > n || j > n {
            return Err(RslabError::IoError(format!(
                "{}: line {}: index ({}, {}) out of bounds for {}x{} matrix",
                source,
                line_no + 1,
                i,
                j,
                n,
                n
            )));
        }

        let (i0, j0) = (i - 1, j - 1);
        if fold_to_lower && i0 < j0 {
            entries.push((j0, i0, v));
        } else {
            entries.push((i0, j0, v));
        }
    }

    if entries.len() != nnz {
        return Err(RslabError::IoError(format!(
            "{}: declared nnz {} does not match the {} entries in the file",
            source,
            nnz,
            entries.len()
        )));
    }

    Ok(MtxMatrix { n, entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_matrices_read_back() {
        let dir = std::env::temp_dir();
        let g =
            GeneralCsc::from_triplets(3, &[0, 2, 1, 0], &[0, 0, 1, 2], &[1.5, -2.0, 1e-300, 3.25])
                .unwrap();
        let path = dir.join(format!("rslab_write_{}.mtx", std::process::id()));
        write_mtx(&path, &g).unwrap();
        let back = read_mtx_any(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let MtxLoaded::General(b) = back else {
            panic!("general file read as symmetric");
        };
        assert_eq!(
            (b.col_ptr, b.row_idx),
            (g.col_ptr.clone(), g.row_idx.clone())
        );
        assert!(b
            .values
            .iter()
            .zip(&g.values)
            .all(|(x, &y)| x.re == y && x.im == 0.0));

        let c = |re, im| Complex::new(re, im);
        let s = CscMatrix::from_triplets(
            2,
            &[0, 1, 1],
            &[0, 0, 1],
            &[c(1.0, 0.5), c(-0.1, 2.0), c(3.0, 0.0)],
        )
        .unwrap();
        let text = format_mtx_symmetric(&s);
        let back = parse_mtx_complex(&text, "mem").unwrap().to_csc().unwrap();
        assert_eq!(back.values, s.values);

        let b = format_mtx_array(&[1.0f64, -0.5]);
        assert_eq!(
            b,
            "%%MatrixMarket matrix array real general\n2 1\n1e0\n-5e-1\n"
        );
    }

    #[test]
    fn test_parse_symmetric_3x3() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 2.0
2 1 -1.0
2 2 3.0
3 3 1.5
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 4);

        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 2.0);
        assert_eq!(dense.get(1, 0), -1.0);
        assert_eq!(dense.get(0, 1), -1.0);
        assert_eq!(dense.get(1, 1), 3.0);
        assert_eq!(dense.get(2, 2), 1.5);
        assert_eq!(dense.get(2, 0), 0.0);
    }

    #[test]
    fn huge_nnz_header_does_not_abort() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 100000000000000000
1 1 2.0
2 1 -1.0
";
        let err = parse_mtx(mtx, "test")
            .expect_err("a bogus huge nnz must return a recoverable Err, not abort the process");
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("declared nnz") && msg.contains("does not match"),
                "expected an nnz-count-mismatch error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    #[test]
    fn declared_nnz_must_match_entry_count() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 2.0
2 2 3.0
";
        let err = parse_mtx(mtx, "test").expect_err(
            "a declared nnz that disagrees with the actual entry count must be rejected",
        );
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("declared nnz") && msg.contains("does not match"),
                "expected an nnz-count-mismatch error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_entries_summed_consistently_by_both_paths() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 3
1 1 2.0
1 1 1.0
2 2 5.0
";
        let m = parse_mtx(mtx, "test").expect("valid file with a duplicate coordinate");

        let dense = m.to_dense();
        assert_eq!(
            dense.get(0, 0),
            3.0,
            "to_dense must sum duplicate (1,1) = 2.0 + 1.0"
        );
        assert_eq!(dense.get(1, 1), 5.0);

        let csc = m.to_csc().expect("to_csc");
        let mut v00 = None;
        for k in csc.col_ptr[0]..csc.col_ptr[1] {
            if csc.row_idx[k] == 0 {
                v00 = Some(csc.values[k]);
            }
        }
        assert_eq!(
            v00,
            Some(3.0),
            "to_csc must sum duplicate (1,1); both paths must agree"
        );
    }

    #[test]
    fn multispace_banner_is_accepted() {
        let mtx = "\
%%MatrixMarket   matrix    coordinate real   symmetric
2 2 2
1 1 4.0
2 2 5.0
";
        let m = parse_mtx(mtx, "test")
            .expect("a banner separated by multiple spaces is legal and must parse");
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 2);
    }

    #[test]
    fn tab_separated_banner_is_accepted() {
        let mtx = "%%MatrixMarket\tmatrix\tcoordinate\treal\tsymmetric\n2 2 1\n1 1 7.0\n";
        let m = parse_mtx(mtx, "test").expect("a tab-separated banner is legal and must parse");
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 1);
    }

    #[test]
    fn nan_value_is_rejected() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 nan
";
        let err = parse_mtx(mtx, "test")
            .expect_err("a NaN entry value must be rejected, not read through");
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("non-finite"),
                "expected a non-finite error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    #[test]
    fn inf_value_is_rejected() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 inf
";
        let err = parse_mtx(mtx, "test")
            .expect_err("an inf entry value must be rejected, not read through");
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("non-finite"),
                "expected a non-finite error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_with_comments() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
% This is a comment
% Another comment
2 2 1
1 1 5.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0], (0, 0, 5.0));
    }

    #[test]
    fn test_parse_scientific_notation() {
        let values = ["1.23456789012345678e+02", "-9.87654321098765432e-03"];
        let mtx = format!(
            "%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 {}\n2 1 {}\n",
            values[0], values[1]
        );
        let m = parse_mtx(&mtx, "test").unwrap();
        assert_eq!(m.entries.len(), 2);
        for (entry, text) in m.entries.iter().zip(values) {
            assert_eq!(entry.2, text.parse::<f64>().unwrap());
        }
    }

    #[test]
    fn test_upper_triangle_normalized() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 2 7.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0], (1, 0, 7.0));
    }

    #[test]
    fn test_reject_general_format() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
2 2 1
1 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn test_real_parser_rejects_complex_header() {
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
2 2 1
1 1 1.0 0.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn complex_parser_reads_complex_symmetric() {
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
3 3 4
1 1 2.0 1.0
2 1 -1.0 0.3
2 2 3.0 -0.5
1 3 0.5 0.2
";
        let m = parse_mtx_complex(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 4);
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), Complex::new(2.0, 1.0));
        assert_eq!(dense.get(1, 0), Complex::new(-1.0, 0.3));
        assert_eq!(dense.get(0, 1), Complex::new(-1.0, 0.3));
        assert_eq!(dense.get(2, 0), Complex::new(0.5, 0.2));
    }

    #[test]
    fn complex_parser_rejects_real_header() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 1.0
";
        let err = parse_mtx_complex(mtx, "test").unwrap_err();
        assert!(format!("{}", err).contains("unsupported header"));
    }

    #[test]
    fn complex_parser_rejects_wrong_token_count() {
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
2 2 1
1 1 1.0
";
        let err = parse_mtx_complex(mtx, "test").unwrap_err();
        assert!(format!("{}", err).contains("expected 'i j re im'"));
    }

    #[test]
    fn complex_parser_rejects_non_finite() {
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
2 2 1
1 1 1.0 nan
";
        let err = parse_mtx_complex(mtx, "test").unwrap_err();
        assert!(format!("{}", err).contains("non-finite"));
    }

    #[test]
    fn complex_mtx_round_trips_through_solver() {
        use crate::LdltSolver;
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
3 3 5
1 1 4.0 1.0
2 1 -1.0 0.2
2 2 4.0 1.0
3 2 -1.0 0.2
3 3 4.0 1.0
";
        let a = parse_mtx_complex(mtx, "test").unwrap().to_csc().unwrap();
        let b = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.5),
        ];
        let solver = LdltSolver::factor(&a, &crate::SolverSettings::default()).unwrap();
        let x = solver.solve(&b).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); 3];
        a.symv(&x, &mut ax);
        let res = (0..3).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        let mut abs = a.clone();
        abs.values = a
            .values
            .iter()
            .map(|v| Complex::new(v.norm(), 0.0))
            .collect();
        let mut row_sums = vec![Complex::new(0.0, 0.0); 3];
        abs.symv(&[Complex::new(1.0, 0.0); 3], &mut row_sums);
        let max = |v: &[Complex<f64>]| v.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let omega = res / (max(&row_sums) * max(&x) + max(&b));
        assert!(
            omega <= crate::test_constants::solve_backward_error(3, crate::test_constants::UC64),
            "backward error {omega}"
        );
    }

    #[test]
    fn test_reject_array_format() {
        let mtx = "\
%%MatrixMarket matrix array real symmetric
2 2
1.0
2.0
3.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn test_reject_nonsquare() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 4 1
1 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("square"), "got: {}", msg);
    }

    #[test]
    fn test_reject_out_of_bounds() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
3 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("out of bounds"), "got: {}", msg);
    }

    #[test]
    fn test_empty_matrix() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 0);

        let dense = m.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(dense.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_diagonal_only() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 3
1 1 1.0
2 2 2.0
3 3 3.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 1.0);
        assert_eq!(dense.get(1, 1), 2.0);
        assert_eq!(dense.get(2, 2), 3.0);
        assert_eq!(dense.get(1, 0), 0.0);
    }

    #[test]
    fn test_negative_values() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 3
1 1 -1.0
2 1 -0.0
2 2 -3.5
";
        let m = parse_mtx(mtx, "test").unwrap();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), -1.0);
        assert_eq!(dense.get(1, 1), -3.5);
    }
}
