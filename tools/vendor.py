"""Vendor the rslab library into another repository.

    python tools/vendor.py <dest> [--rev REV]

Writes the library at commit REV (default HEAD) to <dest> (typically
`vendor/rslab` of the consuming repository): the tracked sources of `src/`
without `src/bin`, the four ordering crates without their tests and
examples, the license files, a manifest trimmed to the library (the
upstream features and dependencies, no benches, tests, examples or
dev-dependencies) with its own `[workspace]` so the consumer's workspace
does not build rslab's, and `VENDOR.md` with the provenance. Files of an
earlier vendoring are replaced; `<dest>/Cargo.lock` is kept for cargo to
update.
"""

import argparse
import io
import pathlib
import re
import shutil
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATES = ["rslab-ordering-core", "rslab-amd", "rslab-amf", "rslab-metis"]
LICENSES = ["LICENSE", "NOTICE", "LICENSE-THIRD-PARTY"]
DROP = ["src/bin"] + [f"crates/{c}/{d}" for c in CRATES for d in ("tests", "examples", "benches")]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def section(manifest, name):
    """Body of the top-level `[name]` table (up to the next header)."""
    m = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[|\Z)", manifest, re.M | re.S)
    return m.group(1).strip() if m else ""


def field(package, key):
    m = re.search(rf"^{key}\s*=\s*(.+)$", package, re.M)
    return m.group(1).strip()


def manifest(upstream, rev):
    package = section(upstream, "package")
    fields = "\n".join(
        f"{k} = {field(package, k)}"
        for k in ("name", "version", "edition", "license", "description", "authors", "repository")
    )
    members = ",\n".join(f'    "{m}"' for m in ["."] + [f"crates/{c}" for c in CRATES])
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
{section(upstream, "dependencies")}
"""


def vendor_md(rev, subject, version, dest_rel):
    return f"""# Vendored rslab

Library copy of the rslab sparse direct solver
(https://github.com/milanofthe/rslab).

- Vendored from: commit `{rev}` (rslab {version}), "{subject}"
- Contents: `src/` (library only) plus the ordering crates
  `crates/{{{','.join(CRATES)}}}`, the license files, and a manifest trimmed
  to the library.
- Local changes: none. Do not patch this tree; fix upstream and resync.

## Resync

From a checkout of rslab next to this repository:

```sh
python ../rslab/tools/vendor.py {dest_rel} --rev <commit>
```

then build and test this repository (cargo updates `Cargo.lock`).
"""


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
    (dest / "Cargo.toml").write_text(manifest(upstream, rev), encoding="utf-8")
    try:
        dest_rel = dest.relative_to(pathlib.Path.cwd()).as_posix()
    except ValueError:
        dest_rel = "vendor/rslab"
    (dest / "VENDOR.md").write_text(vendor_md(rev, subject, version, dest_rel), encoding="utf-8")
    print(f"vendored rslab {version} ({rev}) into {dest}")


if __name__ == "__main__":
    main()
