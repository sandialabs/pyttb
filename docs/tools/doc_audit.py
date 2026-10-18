"""Audit rendered pyttb docstrings against docs/source/developer_guide.rst.

Usage: python docs/tools/doc_audit.py [--summary] [module-prefix ...]
Exit code 1 if any findings.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
import sys

WHOLE_MODULES = [
    "pyttb.pyttb_utils",
    "pyttb.gcp.fg",
    "pyttb.gcp.fg_est",
    "pyttb.gcp.fg_setup",
    "pyttb.gcp.handles",
    "pyttb.gcp.optimizers",
    "pyttb.gcp.samplers",
]
CLASSES = [
    "pyttb:tensor",
    "pyttb:sptensor",
    "pyttb:ktensor",
    "pyttb:ttensor",
    "pyttb:sumtensor",
    "pyttb:tenmat",
    "pyttb:sptenmat",
    "pyttb.create_problem:BaseProblem",
    "pyttb.create_problem:CPProblem",
    "pyttb.create_problem:TuckerProblem",
    "pyttb.create_problem:ExistingSolution",
    "pyttb.create_problem:ExistingCPSolution",
    "pyttb.create_problem:ExistingTuckerSolution",
    "pyttb.create_problem:MissingData",
]
FUNCTIONS = [
    "pyttb:tendiag",
    "pyttb:teneye",
    "pyttb:tenones",
    "pyttb:tenrand",
    "pyttb:tenzeros",
    "pyttb:sptendiag",
    "pyttb:sptenrand",
    "pyttb:khatrirao",
    "pyttb:cp_als",
    "pyttb:cp_apr",
    "pyttb:gcp_opt",
    "pyttb:hosvd",
    "pyttb:tucker_als",
    "pyttb.create_problem:create_problem",
    "pyttb:import_data",
    "pyttb:export_data",
    "pyttb.matlab.matlab_support:matlab_print",
    "pyttb:ignore_warnings",
]
SKIP_MEMBERS = {
    "__init__",
    "__deepcopy__",
    "order",
    "__post_init__",
    "__replace__",
    "__new__",
}
EXAMPLES_EXEMPT = re.compile(
    r"^pyttb\.gcp\.(handles\.|optimizers\.\w+\.(update_step|set_failed_epoch|solve)$)"
)


def resolve(spec):
    module, attr = spec.split(":")
    return getattr(importlib.import_module(module), attr)


def class_members(cls, prefix):
    is_dc = dataclasses.is_dataclass(cls) or hasattr(cls, "_fields")
    for name, raw in vars(cls).items():
        if name in SKIP_MEMBERS or (name.startswith("_") and not name.endswith("__")):
            continue
        if is_dc and name.startswith("__"):
            continue  # generated __repr__/__eq__
        is_prop = isinstance(raw, property)
        func = raw.fget if is_prop else getattr(raw, "__func__", raw)
        if not inspect.isfunction(func):
            continue
        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue  # function-valued dataclass defaults, e.g. randn
        yield f"{prefix}.{name}", func, is_prop


def targets():
    for module_name in WHOLE_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module_name:
                continue
            if inspect.isfunction(obj):
                yield f"{module_name}.{name}", obj, False
            elif inspect.isclass(obj):
                yield from class_members(obj, f"{module_name}.{name}")
    for spec in CLASSES:
        yield from class_members(resolve(spec), spec.replace(":", "."))
    for spec in FUNCTIONS:
        yield spec.replace(":", "."), resolve(spec), False


def has_section(doc, name):
    return re.search(rf"^{name}\n-{{3,}}$", doc, re.M) is not None


def bad_returns(doc, annotation):
    match = re.search(r"^Returns\n-{3,}\n(.+)", doc, re.M)
    if not match:
        return False
    first = match.group(1).rstrip()
    if first == ":" or re.fullmatch(r":class:`[^`]+`", first):
        return True
    if re.fullmatch(r"(bool|int|float|str|list|tuple|dict)", first):
        return True
    is_tuple = str(annotation).startswith("tuple")
    return not is_tuple and re.match(r"^\w+\s*:", first) is not None


def audit():
    findings = []
    for where, func, is_prop in targets():
        doc = inspect.getdoc(func) or ""
        if not doc:
            findings.append(("no-docstring", where))
            continue
        signature = inspect.signature(func)
        params = [p for p in signature.parameters if p not in ("self", "cls")]
        annotation = signature.return_annotation
        returns_value = annotation not in (inspect.Signature.empty, None, "None")
        if params and not is_prop and not has_section(doc, "Parameters"):
            findings.append(("missing-parameters", where))
        if returns_value and not is_prop and not has_section(doc, "Returns"):
            findings.append(("missing-returns", where))
        if is_prop and has_section(doc, "Returns"):
            findings.append(("property-has-returns", where))
        if bad_returns(doc, annotation):
            findings.append(("returns-style", where))
        if ">>>" not in doc and not EXAMPLES_EXEMPT.match(where):
            findings.append(("missing-examples", where))
        if re.search(r"^(Example|Arguments)\n-{3,}$", doc, re.M):
            findings.append(("bad-heading", where))
        if re.match(r"^[A-Z_]{4,}\b", doc):
            findings.append(("matlab-uppercase", where))
        if "\t" in (func.__doc__ or ""):
            findings.append(("tab-in-docstring", where))
    return findings


def main(argv):
    summary = "--summary" in argv
    prefixes = [a for a in argv if not a.startswith("--")]
    findings = [f for f in audit() if not prefixes or f[1].startswith(tuple(prefixes))]
    counts = {}
    for kind, _ in findings:
        counts[kind] = counts.get(kind, 0) + 1
    if not summary:
        for kind, where in sorted(findings):
            print(f"{kind:22} {where}")
    print("SUMMARY", dict(sorted(counts.items())), "total", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
