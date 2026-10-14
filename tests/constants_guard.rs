//! Guard against unnamed numbers: every tuning value, threshold, limit,
//! tolerance and semantic constant lives in the constants module of its
//! crate (`src/constants/`, `crates/*/src/constants.rs`,
//! `python/src/constants.rs`).
//!
//! The test parses every library source of the workspace and of the Python
//! bindings (`src`, `crates/*/src`, `python/src`; test code excluded) and
//! fails on a numeric literal outside a constants module unless
//!
//! - its value is trivial (`0`, `1`, `2`, `0.0`, `0.5`, `1.0`, `2.0`),
//! - it sits in a structural position: a type (`[f64; 3]`), a generic
//!   argument, a pattern (`3 => ..`) or the index of an index expression
//!   (`v[3]`, `v[..4]`, `v[i + 3]`),
//! - its line carries a `// lit: <reason>` comment (a formula's own
//!   number, such as the 6 of a sum of squares), or
//! - its file or item is in the exemption table below (algorithm-defined
//!   constants that belong next to their code).
//!
//! Attributes are skipped except `#[pyo3(signature = ...)]`, whose defaults
//! are what Python callers see and must reference constants. Literals inside
//! macro invocations follow the same rule.

use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

/// Files kept out of the scan entirely, with the reason.
const EXEMPT_FILES: &[(&str, &str)] = &[
    (
        "src/matgen/",
        "test-matrix generators: the numbers are test data",
    ),
    (
        "crates/rslab-amd/src/bin/",
        "benchmark tool, not library code",
    ),
    (
        "crates/rslab-metis/src/rng.rs",
        "SplitMix64: published algorithm constants",
    ),
];

/// Items (`file`, enclosing item) kept out of the scan, with the reason.
const EXEMPT_ITEMS: &[(&str, &str, &str)] = &[
    (
        "src/symbolic/supervariables.rs",
        "FNV_OFFSET",
        "FNV-1a 64: published hash constant",
    ),
    (
        "src/symbolic/supervariables.rs",
        "FNV_PRIME",
        "FNV-1a 64: published hash constant",
    ),
    (
        "src/numeric/ldlt/bunch_kaufman.rs",
        "bk_alpha",
        "Bunch-Kaufman alpha (1 + sqrt(17)) / 8: algorithm-defined",
    ),
];

/// Roots scanned, relative to the workspace root. `crates/*/src` expands.
const ROOTS: &[&str] = &["src", "crates/*/src", "python/src"];

#[derive(Debug, Clone)]
struct Hit {
    file: String,
    line: usize,
    item: String,
    lit: String,
}

#[test]
fn numeric_literals_are_named() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for r in ROOTS {
        if let Some(prefix) = r.strip_suffix("/*/src") {
            let mut dirs: Vec<_> = std::fs::read_dir(root.join(prefix))
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path().join("src")))
                .filter(|p| p.is_dir())
                .collect();
            dirs.sort();
            for d in dirs {
                collect(&d, &mut files);
            }
        } else {
            collect(&root.join(r), &mut files);
        }
    }
    files.sort();

    // Out-of-line modules declared under `#[cfg(test)]` are test code.
    let mut sources = Vec::new();
    let mut test_files = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        let ast = syn::parse_file(&text).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        test_modules(f, &ast.items, &mut test_files);
        sources.push((f.clone(), text, ast));
    }

    let mut hits = Vec::new();
    let mut marked: BTreeMap<String, usize> = BTreeMap::new();
    for (f, text, ast) in &sources {
        let rel = rel_path(root, f);
        if test_files.contains(f)
            || is_constants_module(&rel)
            || EXEMPT_FILES.iter().any(|(p, _)| rel.starts_with(p))
        {
            continue;
        }
        let lines: Vec<&str> = text.lines().collect();
        let mut v = Scan {
            file: rel.clone(),
            lines: &lines,
            ctx: Vec::new(),
            hits: &mut hits,
            marked: marked.entry(rel.clone()).or_default(),
        };
        v.visit_file(ast);
    }

    for (file, n) in &marked {
        if *n > 0 {
            println!("{file}: {n} literal(s) marked '// lit:'");
        }
    }

    let errors: Vec<String> = hits
        .iter()
        .map(|h| {
            format!(
                "{}:{}: literal {} in {} is not named (move it to the crate's constants \
                 module, or mark '// lit: <reason>')",
                h.file, h.line, h.lit, h.item
            )
        })
        .collect();
    assert!(
        errors.is_empty(),
        "{} constants guard violation(s):\n{}",
        errors.len(),
        errors.join("\n")
    );
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if !matches!(name.as_str(), "tests" | "benches" | "examples") {
                collect(&p, out);
            }
        } else if name.ends_with(".rs") && name != "tests.rs" && name != "build.rs" {
            out.push(p);
        }
    }
}

fn rel_path(root: &Path, f: &Path) -> String {
    f.strip_prefix(root)
        .unwrap_or(f)
        .to_string_lossy()
        .replace('\\', "/")
}

fn is_constants_module(rel: &str) -> bool {
    rel.ends_with("/constants.rs") || rel.contains("/constants/")
}

/// Whether the attributes mark test-only code.
fn is_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        let p = a.path();
        if p.is_ident("test") {
            return true;
        }
        if !p.is_ident("cfg") {
            return false;
        }
        let Ok(list) = a.meta.require_list() else {
            return false;
        };
        let toks: Vec<TokenTree> = list.tokens.clone().into_iter().collect();
        match toks.first() {
            Some(TokenTree::Ident(i)) if i == "test" => toks.len() == 1,
            Some(TokenTree::Ident(i)) if i == "all" => match toks.get(1) {
                Some(TokenTree::Group(g)) => g
                    .stream()
                    .into_iter()
                    .any(|t| matches!(&t, TokenTree::Ident(i) if i == "test")),
                _ => false,
            },
            _ => false,
        }
    })
}

/// The files of the out-of-line `#[cfg(test)] mod x;` declarations in `file`.
fn test_modules(file: &Path, items: &[syn::Item], out: &mut Vec<PathBuf>) {
    let name = file.file_stem().unwrap_or_default().to_string_lossy();
    let dir = file.parent().unwrap_or(Path::new("."));
    let base = if matches!(name.as_ref(), "mod" | "lib" | "main") {
        dir.to_path_buf()
    } else {
        dir.join(name.as_ref())
    };
    for it in items {
        if let syn::Item::Mod(m) = it {
            if m.content.is_none() && is_test(&m.attrs) {
                let id = m.ident.to_string();
                out.push(base.join(format!("{id}.rs")));
                out.push(base.join(&id).join("mod.rs"));
            }
        }
    }
}

struct Scan<'a> {
    file: String,
    lines: &'a [&'a str],
    ctx: Vec<String>,
    hits: &'a mut Vec<Hit>,
    marked: &'a mut usize,
}

impl Scan<'_> {
    fn item(&self) -> String {
        if self.ctx.is_empty() {
            "<file>".to_string()
        } else {
            self.ctx.join("::")
        }
    }

    fn exempt(&self) -> bool {
        EXEMPT_ITEMS
            .iter()
            .any(|(f, i, _)| *f == self.file && self.ctx.iter().any(|c| c == i))
    }

    fn check(&mut self, text: String, trivial: bool, line: usize) {
        if trivial || self.exempt() {
            return;
        }
        let src = self.lines.get(line.wrapping_sub(1)).copied().unwrap_or("");
        if let Some(i) = src.find("// lit:") {
            if !src[i + "// lit:".len()..].trim().is_empty() {
                *self.marked += 1;
                return;
            }
        }
        self.hits.push(Hit {
            file: self.file.clone(),
            line,
            item: self.item(),
            lit: text,
        });
    }

    fn check_lit(&mut self, lit: &syn::Lit) {
        match lit {
            syn::Lit::Int(i) => {
                let trivial = matches!(i.base10_parse::<u128>(), Ok(0..=2));
                self.check(i.to_string(), trivial, i.span().start().line);
            }
            syn::Lit::Float(f) => {
                let trivial =
                    matches!(f.base10_parse::<f64>(), Ok(v) if [0.0, 0.5, 1.0, 2.0].contains(&v));
                self.check(f.to_string(), trivial, f.span().start().line);
            }
            _ => {}
        }
    }

    /// Numeric literals of a token stream (macro bodies, pyo3 signatures).
    fn check_tokens(&mut self, ts: TokenStream) {
        for t in ts {
            match t {
                TokenTree::Group(g) => self.check_tokens(g.stream()),
                TokenTree::Literal(l) => {
                    if let Ok(lit) = syn::parse_str::<syn::Lit>(&l.to_string()) {
                        let line = l.span().start().line;
                        match lit {
                            syn::Lit::Int(i) => {
                                let trivial = matches!(i.base10_parse::<u128>(), Ok(0..=2));
                                self.check(i.to_string(), trivial, line);
                            }
                            syn::Lit::Float(f) => {
                                let trivial = matches!(f.base10_parse::<f64>(), Ok(v) if [0.0, 0.5, 1.0, 2.0].contains(&v));
                                self.check(f.to_string(), trivial, line);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn scoped(&mut self, name: String, f: impl FnOnce(&mut Self)) {
        self.ctx.push(name);
        f(self);
        self.ctx.pop();
    }
}

impl Scan<'_> {
    /// An index expression: literals as the index, as range bounds or as an
    /// offset `i + 3` / `n - 1` are structural; anything else is scanned.
    fn visit_index(&mut self, e: &syn::Expr) {
        match e {
            syn::Expr::Lit(_) => {}
            syn::Expr::Range(r) => {
                for b in [&r.start, &r.end].into_iter().flatten() {
                    self.visit_index(b);
                }
            }
            syn::Expr::Binary(b) if matches!(b.op, syn::BinOp::Add(_) | syn::BinOp::Sub(_)) => {
                for side in [&*b.left, &*b.right] {
                    if !matches!(side, syn::Expr::Lit(_)) {
                        self.visit_expr(side);
                    }
                }
            }
            _ => self.visit_expr(e),
        }
    }
}

fn type_name(t: &syn::Type) -> String {
    match t {
        syn::Type::Path(p) => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default(),
        _ => "impl".to_string(),
    }
}

impl<'ast> Visit<'ast> for Scan<'_> {
    fn visit_attribute(&mut self, a: &'ast syn::Attribute) {
        if a.path().is_ident("pyo3") {
            if let Ok(list) = a.meta.require_list() {
                self.check_tokens(list.tokens.clone());
            }
        }
    }

    fn visit_type(&mut self, _: &'ast syn::Type) {}

    fn visit_pat(&mut self, _: &'ast syn::Pat) {}

    fn visit_generic_argument(&mut self, _: &'ast syn::GenericArgument) {}

    fn visit_lit(&mut self, l: &'ast syn::Lit) {
        self.check_lit(l);
    }

    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        self.check_tokens(m.tokens.clone());
    }

    fn visit_expr_index(&mut self, e: &'ast syn::ExprIndex) {
        for a in &e.attrs {
            self.visit_attribute(a);
        }
        self.visit_expr(&e.expr);
        self.visit_index(&e.index);
    }

    fn visit_item(&mut self, i: &'ast syn::Item) {
        let attrs = match i {
            syn::Item::Const(x) => &x.attrs,
            syn::Item::Enum(x) => &x.attrs,
            syn::Item::Fn(x) => &x.attrs,
            syn::Item::Impl(x) => &x.attrs,
            syn::Item::Macro(x) => &x.attrs,
            syn::Item::Mod(x) => &x.attrs,
            syn::Item::Static(x) => &x.attrs,
            syn::Item::Struct(x) => &x.attrs,
            syn::Item::Trait(x) => &x.attrs,
            syn::Item::Use(x) => &x.attrs,
            _ => return visit::visit_item(self, i),
        };
        if !is_test(attrs) {
            visit::visit_item(self, i);
        }
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.scoped(i.sig.ident.to_string(), |s| visit::visit_item_fn(s, i));
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        self.scoped(i.ident.to_string(), |s| visit::visit_item_const(s, i));
    }

    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        self.scoped(i.ident.to_string(), |s| visit::visit_item_static(s, i));
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.scoped(i.ident.to_string(), |s| visit::visit_item_enum(s, i));
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.scoped(i.ident.to_string(), |s| visit::visit_item_struct(s, i));
    }

    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        self.scoped(i.ident.to_string(), |s| visit::visit_item_mod(s, i));
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.scoped(i.ident.to_string(), |s| visit::visit_item_trait(s, i));
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        let name = i
            .ident
            .as_ref()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "macro".to_string());
        self.scoped(name, |s| visit::visit_item_macro(s, i));
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        self.scoped(type_name(&i.self_ty), |s| visit::visit_item_impl(s, i));
    }

    fn visit_impl_item(&mut self, i: &'ast syn::ImplItem) {
        let skip = match i {
            syn::ImplItem::Fn(f) => is_test(&f.attrs),
            syn::ImplItem::Const(c) => is_test(&c.attrs),
            _ => false,
        };
        if !skip {
            visit::visit_impl_item(self, i);
        }
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        self.scoped(i.sig.ident.to_string(), |s| visit::visit_impl_item_fn(s, i));
    }

    fn visit_impl_item_const(&mut self, i: &'ast syn::ImplItemConst) {
        self.scoped(i.ident.to_string(), |s| visit::visit_impl_item_const(s, i));
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        if !is_test(&i.attrs) {
            self.scoped(i.sig.ident.to_string(), |s| {
                visit::visit_trait_item_fn(s, i)
            });
        }
    }

    fn visit_trait_item_const(&mut self, i: &'ast syn::TraitItemConst) {
        self.scoped(i.ident.to_string(), |s| visit::visit_trait_item_const(s, i));
    }

    fn visit_stmt(&mut self, s: &'ast syn::Stmt) {
        let test = match s {
            syn::Stmt::Local(l) => is_test(&l.attrs),
            syn::Stmt::Macro(m) => is_test(&m.attrs),
            _ => false,
        };
        if !test {
            visit::visit_stmt(self, s);
        }
    }

    fn visit_expr(&mut self, e: &'ast syn::Expr) {
        let attrs: &[syn::Attribute] = match e {
            syn::Expr::Block(b) => &b.attrs,
            syn::Expr::Call(c) => &c.attrs,
            syn::Expr::MethodCall(c) => &c.attrs,
            _ => &[],
        };
        if !is_test(attrs) {
            visit::visit_expr(self, e);
        }
    }
}
