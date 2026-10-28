"""Vendor the rslab library into another repository.

    python tools/vendor.py <dest> [--rev REV]

Writes the library at commit REV (default HEAD) to <dest> (typically
`vendor/rslab` of the consuming repository): the tracked sources of `src/`
without `src/bin`, the four ordering crates without their tests and
examples, the license files, a manifest trimmed to the library (the
upstream features and dependencies, no benches, tests, examples or
dev-dependencies) with its own `[workspace]` so the consumer's workspace
does not build rslab's, `VENDOR.md` with the provenance, and a
`Cargo.lock` that matches the trimmed manifest: upstream's lock at REV,
pruned by cargo to the library's dependencies, so the pinned versions are
the ones tested upstream and no later cargo run in <dest> (a consumer's
`maturin sdist` runs `cargo metadata` there) rewrites it. Files of an
earlier vendoring are replaced.

A dependency upstream takes from git (`name = { git = .., rev = .. }`) is
vendored along at that revision into `crates/<name>` (its library, README
and license, no tests, examples or dev-dependencies) and taken by path, so
the consumer builds without fetching it. Its source is a checkout of it
next to rslab where that holds the revision, else a clone of its
repository.
"""

import argparse
import io
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATES = ["rslab-ordering-core", "rslab-amd", "rslab-amf", "rslab-metis"]
LICENSES = ["LICENSE", "NOTICE", "LICENSE-THIRD-PARTY"]
DROP = ["src/bin"] + [f"crates/{c}/{d}" for c in CRATES for d in ("tests", "examples", "benches")]


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


# One dependency line `name = { ..., git = "..", ... }`.
GIT_DEP = re.compile(r"^(?P<name>[\w-]+)\s*=\s*\{(?P<body>[^}\n]*\bgit\s*=[^}\n]*)\}", re.M)


def git_deps(dependencies):
    """`(name, url, rev)` of every dependency taken from git."""
    out = []
    for m in GIT_DEP.finditer(dependencies):
        url = re.search(r'\bgit\s*=\s*"([^"]+)"', m["body"]).group(1)
        rev = re.search(r'\brev\s*=\s*"([^"]+)"', m["body"])
        if rev is None:
            raise SystemExit(f"vendor: the git dependency {m['name']} has no rev")
        out.append((m["name"], url, rev.group(1)))
    return out


def by_path(dependencies, names):
    """`dependencies` with the git source of each of `names` replaced by its
    vendored path."""

    def sub(m):
        if m["name"] not in names:
            return m[0]
        keys = [k.strip() for k in re.split(r",(?![^\[]*\])", m["body"])]
        keys = [k for k in keys if k and not re.match(r"(git|rev)\s*=", k)]
        return f'{m["name"]} = {{ path = "crates/{m["name"]}", {", ".join(keys)} }}'

    return GIT_DEP.sub(sub, dependencies)


def vendor_git_dep(dest, name, url, rev):
    """The library of the git dependency `name` at `rev` into
    `<dest>/crates/<name>`; returns its short commit."""
    with tempfile.TemporaryDirectory() as tmp:
        src = ROOT.parent / name
        try:
            git("cat-file", "-e", f"{rev}^{{commit}}", cwd=src)
        except (subprocess.CalledProcessError, OSError):
            src = pathlib.Path(tmp) / name
            subprocess.run(["git", "clone", "--quiet", url, str(src)], check=True)
        short = git("rev-parse", "--short", f"{rev}^{{commit}}", cwd=src).decode().strip()
        top = git("ls-tree", "--name-only", rev, cwd=src).decode().split()
        keep = [f for f in top if f in ("src", "Cargo.toml", "README.md") or f.startswith("LICENSE")]
        tar = git("archive", "--format=tar", rev, *keep, cwd=src)
    target = dest / "crates" / name
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(target, filter="data")
    crate = (target / "Cargo.toml").read_text(encoding="utf-8")
    crate = re.sub(r"^\[dev-dependencies\]\n.*?(?=^\[|\Z)", "", crate, flags=re.M | re.S)
    (target / "Cargo.toml").write_text(crate, encoding="utf-8")
    return short


def section(manifest, name):
    """Body of the top-level `[name]` table (up to the next header)."""
    m = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[|\Z)", manifest, re.M | re.S)
    return m.group(1).strip() if m else ""


def field(package, key):
    m = re.search(rf"^{key}\s*=\s*(.+)$", package, re.M)
    return m.group(1).strip()


def manifest(upstream, rev, vendored):
    package = section(upstream, "package")
    fields = "\n".join(
        f"{k} = {field(package, k)}"
        for k in ("name", "version", "edition", "license", "description", "authors", "repository")
    )
    members = ",\n".join(f'    "{m}"' for m in ["."] + [f"crates/{c}" for c in CRATES + vendored])
    return f"""# Vendored copy of rslab ({field(package, 'repository').strip('"')}) at commit
# {rev}, written by rslab's tools/vendor.py (see VENDOR.md). The upstream
# manifest trimmed to the library: its features and dependencies, no
# benches, tests, examples or dev-dependencies. The own [workspace] keeps
# the vendored tree out of the consuming workspace.
[workspace]
members = [
{members},
]
resolver = "2"

[package]
{fields}

[lib]
name = "rslab"
path = "src/lib.rs"

[features]
{section(upstream, "features")}

[dependencies]
{by_path(section(upstream, "dependencies"), vendored)}
"""


def vendor_md(rev, subject, version, dest_rel, along):
    return f"""# Vendored rslab

Library copy of the rslab sparse direct solver
(https://github.com/milanofthe/rslab).

- Vendored from: commit `{rev}` (rslab {version}), "{subject}"
- Contents: `src/` (library only) plus the ordering crates
  `crates/{{{','.join(CRATES)}}}`, the license files, and a manifest trimmed
  to the library.{along}
- Local changes: none. Do not patch this tree; fix upstream and resync.

## Resync

From a checkout of rslab next to this repository:

```sh
python ../rslab/tools/vendor.py {dest_rel} --rev <commit>
```

then build and test this repository. The vendored `Cargo.lock` is upstream's,
pruned to the library; the consumer's own `Cargo.lock` is updated by cargo.
"""


def lock(dest, rev):
    """`<dest>/Cargo.lock`: upstream's lock at `rev`, pruned by cargo to the
    trimmed manifest (the dev-dependencies, benches and binaries are gone).
    Pruning resolves nothing new, so the versions stay upstream's."""
    (dest / "Cargo.lock").write_bytes(git("show", f"{rev}:Cargo.lock"))
    subprocess.run(["cargo", "metadata", "--format-version", "1", "--offline"],
                   cwd=dest, check=True, stdout=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dest", type=pathlib.Path)
    p.add_argument("--rev", default="HEAD")
    a = p.parse_args()
    rev = git("rev-parse", "--short", f"{a.rev}^{{commit}}").decode().strip()
    subject = git("log", "-1", "--format=%s", rev).decode().strip()
    upstream = git("show", f"{rev}:Cargo.toml").decode()
    dest = a.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for old in ["src", "crates", "README.md", *LICENSES]:
        target = dest / old
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    tar = git("archive", "--format=tar", rev, "src", *[f"crates/{c}" for c in CRATES], *LICENSES)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="data")
    for d in DROP:
        shutil.rmtree(dest / d, ignore_errors=True)
    for c in CRATES:
        crate = (dest / "crates" / c / "Cargo.toml").read_text(encoding="utf-8")
        crate = re.sub(r"^\[dev-dependencies\]\n.*?(?=^\[|\Z)", "", crate, flags=re.M | re.S)
        (dest / "crates" / c / "Cargo.toml").write_text(crate, encoding="utf-8")
    version = field(section(upstream, "package"), "version").strip('"')
    deps = git_deps(section(upstream, "dependencies"))
    along = "".join(
        f"\n- Vendored along: `crates/{name}` from {url} at commit "
        f"`{vendor_git_dep(dest, name, url, drev)}`, taken by path."
        for name, url, drev in deps
    )
    (dest / "Cargo.toml").write_text(manifest(upstream, rev, [d[0] for d in deps]), encoding="utf-8")
    try:
        dest_rel = dest.relative_to(pathlib.Path.cwd()).as_posix()
    except ValueError:
        dest_rel = "vendor/rslab"
    (dest / "VENDOR.md").write_text(vendor_md(rev, subject, version, dest_rel, along), encoding="utf-8")
    lock(dest, rev)
    print(f"vendored rslab {version} ({rev}) into {dest}")


if __name__ == "__main__":
    main()
