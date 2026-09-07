#!/usr/bin/env python3
"""Generate docs/api.md from the runtime docstrings of the rslab package.

The compiled extension carries the docstrings of the classes and methods
(written next to the Rust code), the Python package those of the wrapper
functions; this script renders both into one Markdown reference so the
reference cannot drift from the code. `--check` exits non-zero when the
committed file is stale (run by the test suite).
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rslab  # noqa: E402

SECTIONS = [
    ("One-shot solve", ["spsolve"]),
    ("Factor handles", ["ldlt", "lu", "klu", "Ldlt", "Lu", "Klu"]),
    ("Symbolic analysis", ["analyze", "LdltSymbolic", "LuSymbolic", "KluSymbolic"]),
    ("Configuration", ["Settings", "KluSettings", "Interrupt"]),
    ("Krylov solvers", ["gmres", "gmres_block", "cocg", "cocr", "KrylovResult", "Recycle"]),
    ("Machine calibration and logging", ["install_diagnose", "set_log_level", "log_level", "set_log_sink"]),
]


def _doc(obj) -> str:
    return inspect.cleandoc(obj.__doc__ or "")


def _signature(name: str, obj) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        text = getattr(obj, "__text_signature__", None)
        sig = text.replace("($self, ", "(").replace("($self)", "()") if text else "(...)"
    sig = sig.replace("(self, /, ", "(").replace("(self, /)", "()")
    return f"{name}{sig}"


ENTRY_SECTIONS = {"Parameters", "Returns", "Raises", "Attributes", "Yields", "Other Parameters"}


def _render_rst(doc: str) -> str:
    """Light RST-to-Markdown: code blocks, math, roles, section underlines."""
    out = []
    lines = doc.splitlines()
    i = 0
    section = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped in (".. code-block:: python", ".. math::"):
            lang = "python" if "code-block" in stripped else "math"
            i += 1
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
                block.append(lines[i])
                i += 1
            code = textwrap.dedent("\n".join(block)).strip("\n")
            out.append(f"\x00FENCE\x00{lang}\n{code}\n\x00FENCE\x00")
            continue
        # Numpydoc section headers: a title line followed by dashes.
        if i + 1 < len(lines) and lines[i + 1].strip() and set(lines[i + 1].strip()) == {"-"} and stripped:
            section = stripped
            out.append(f"**{stripped}**")
            out.append("")
            i += 2
            continue
        # Entries of a parameter-style section: "name : type" (or a bare
        # type for Returns / Raises) at column 0, indented description below.
        if section in ENTRY_SECTIONS and stripped and not line.startswith(" ") and not stripped.startswith("."):
            name, sep, typ = stripped.partition(" : ")
            out.append(f"- `{name}` ({typ}):" if sep else f"- `{name}`:")
            i += 1
            continue
        if section in ENTRY_SECTIONS and line.startswith("    ") and out and out[-1].startswith("- `"):
            out[-1] += " " + stripped
            i += 1
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    for role in (":func:", ":class:", ":meth:", ":attr:", ":math:", ":doi:"):
        text = text.replace(role, "")
    return text.replace("``", "`").replace("\x00FENCE\x00", "```")


def _members(cls):
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") and name not in ("__iter__",):
            continue
        if name in ("__iter__",):
            continue
        if inspect.isroutine(member) or isinstance(member, property) or hasattr(member, "__get__"):
            yield name, member


def render() -> str:
    parts = [
        "# rslab Python API reference",
        "",
        f"Generated from the docstrings of `rslab` {rslab.__version__} by "
        "`tools/gen_api_reference.py`; do not edit by hand.",
        "",
        "## Package",
        "",
        _render_rst(_doc(rslab)),
        "",
    ]
    for title, names in SECTIONS:
        parts += [f"## {title}", ""]
        for name in names:
            obj = getattr(rslab, name)
            if inspect.isclass(obj):
                parts += [f"### class `{name}`", "", _render_rst(_doc(obj)), ""]
                attrs, methods = [], []
                for mname, member in _members(obj):
                    if isinstance(member, property) or type(member).__name__ == "getset_descriptor":
                        attrs.append((mname, member))
                    elif callable(member):
                        methods.append((mname, member))
                if attrs:
                    parts += ["**Attributes**", ""]
                    for mname, member in attrs:
                        d = _doc(member).replace("\n", " ")
                        parts.append(f"- `{mname}`: {d}" if d else f"- `{mname}`")
                    parts.append("")
                for mname, member in methods:
                    parts += [f"#### `{_signature(f'{name}.{mname}', member)}`", "", _render_rst(_doc(member)), ""]
            else:
                parts += [f"### `{_signature(name, obj)}`", "", _render_rst(_doc(obj)), ""]
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if docs/api.md is stale")
    args = ap.parse_args()
    target = ROOT / "docs" / "api.md"
    text = render()
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != text:
            print(f"{target} is stale; run `python tools/gen_api_reference.py`")
            return 1
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
