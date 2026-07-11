---
name: fedoo-doc
description: >-
  Review and enforce fedoo's Python documentation convention: interface/conceptual doc
  lives in NumPy-style docstrings (rendered by Sphinx autodoc/napoleon), theory and
  tutorials go in docs/*.rst or the examples gallery, and inline code comments are kept
  to short, load-bearing implementation notes only. Explicitly hunts review-artifact
  comments ("fixed", "now correctly", "previously this failed because…") — they are
  noise and must be deleted. Use when the user asks to review/clean up/audit comments
  or docstrings, check Sphinx compliance, or document new code. Produces a read-only
  report grouped by action; applies fixes only when explicitly asked (e.g. "--fix").
---

# fedoo-doc — Python documentation & comment-placement reviewer

Enforce fedoo's house style for code documentation:

- **Docstring = the contract.** Conceptual and interface documentation goes in
  **NumPy-style docstrings** (Sphinx `napoleon` renders them via autodoc/autosummary).
  This is what `docs/` builds.
- **Sphinx pages = the theory.** Longer conceptual material (formulations, conventions,
  math derivations, how-to guides) belongs in `docs/*.rst` or as a gallery example under
  `examples/` — **not** as a wall of prose in a docstring, and never as inline comments.
- **Code body = the implementation.** Inline comments are **short, very concise** notes
  on *subtle implementation points only* — never a restatement of the API, the model,
  or the change history.

## Arguments / scope

The argument (if any) selects what to review. Default to the **uncommitted/branch diff**
when there is one, else ask.

- *(no arg)* → review files changed on the current branch
  (`git diff --name-only master...HEAD` + working tree). If the tree is clean and there
  is no branch diff, ask the user for a scope before scanning all of `fedoo/`.
- `diff` → only changed `.py` files (same as default).
- a path/dir/glob (e.g. `fedoo/constraint`, `mean_motion`) → restrict to matching files.
- `all` → full sweep of `fedoo/` (large; warn it is a big read).
- `--fix` (combinable with any scope) → after reporting, apply only the **safe** edits
  (buckets A & B below), pausing for confirmation on anything destructive. Never
  auto-apply bucket E.

When reviewing a function, read its docstring *and* its body together. A comment is
"redundant" only if the docstring (or the code itself) already carries the information —
verify, don't assume.

## The convention, precisely

### Docstrings — NumPy style (target style)

Reference examples already at target style: `fedoo/core/mesh.py` (`Mesh` class),
`fedoo/core/assembly.py`. Follow that pattern:

```python
class Mesh(MeshBase):
    """Fedoo Mesh object.

    A fedoo mesh can be initialized with the following constructors:

      * :py:meth:`fedoo.Mesh.read` to load a mesh from a file ...

    Parameters
    ----------
    nodes: numpy array of float
        List of nodes coordinates. nodes[i] is the coordinate of the ith node.
    elm_type: str
        Type of the element. ...

    Example
    -------
    >>> import fedoo as fd
    >>> mesh = fd.Mesh(nodes, elm, 'quad4', ndim=3, name='unit square mesh')
    """
```

Rules for docstrings:
- One-line summary first, then optional extended description, then the NumPy sections
  actually needed: `Parameters`, `Returns`, `Attributes`, `Notes`, `Example`/`Examples`,
  `See Also`. Don't add empty boilerplate sections.
- **Math** goes in `Notes` using rst: inline ``:math:`...``` or a ``.. math::`` block.
  State units, sign conventions, Voigt/vector ordering, and DOF conventions **here**
  (or in a `docs/*.rst` page it links to), not in the function body.
- Cross-reference with ``:py:class:`fedoo.X```, ``:py:meth:`...```, and `:ref:` targets —
  fedoo's docs already use these; keep them valid.
- Every **public** class/function/method should have a docstring. Private helpers
  (`_foo`) get a one-line docstring or nothing — not a full NumPy block.
- Runnable, tutorial-sized demonstrations belong in `examples/` (the sphinx-gallery
  picks up every `.py` there); docstring `Example` sections stay minimal.

### Code body — minimal implementation comments (target style)

An inline comment earns its place **only** if it states something the code itself can't
show, e.g.:
- a non-obvious sign/ordering/convention choice (`# D = -R(U_curr): DiffOp coef is the
  OPPOSITE of the physical residual`);
- a numerical guard or robustness policy and *why* it's needed;
- a load-bearing algebraic identity or index-layout fact not visible in the call;
- a citation for a specific formula.

Everything else is noise to remove. In particular — and this is the #1 target of this
skill — **review-artifact comments**:

- `# fix for the bug where ...` / `# this was previously wrong because ...`
- `# now correctly handles X` / `# as pointed out in review ...`
- `# use Y instead of Z to avoid the issue with ...` when Y is simply the correct code
- multi-line justifications of why the change is right, addressed to a reviewer

These document the *diff*, not the *code*. They are noise the moment the PR merges —
git history and the PR discussion already record what was wrong. **Delete them.** If the
underlying constraint is genuinely non-obvious and permanent, compress it to one line
stating the constraint itself (not its history): keep "`# t_fact must stay frozen during
NR sub-iterations`", delete "`# previously t_fact was updated here which caused ...`".

## What to look for (classification buckets)

Group findings by **ACTION**, not by file — that is how they get executed:

- **A — DELETE dead code.** Commented-out blocks, `# print(...)` debug leftovers,
  abandoned alternatives. Highest value, ~zero risk. **Large deletes (tens of lines)
  must be listed explicitly and confirmed before removal.**
- **B — DELETE noise comments.** Review-artifact comments (see above), narration of the
  next line (`# loop over elements`), and restatements of the docstring/API. Low risk —
  but check the comment isn't the only place a real constraint is recorded.
- **C — MOVE / PROMOTE (net-new doc value).** Real explanation living only in the code
  body that belongs in the contract: missing docstrings on public API, conventions or
  units stated only inline, theory blocks that should become a `Notes` section or a
  `docs/*.rst` page, a worked usage that should become a gallery example. This *adds*
  documentation; keep the NumPy/rst style above.
- **D — KEEP (judgment).** Load-bearing "why *this* code" rationale. Do **not** strip;
  at most trim the historical part. When unsure, keep.
- **E — BUGS surfaced (DO NOT auto-edit).** Doc work routinely trips over real defects:
  docstrings whose parameters disagree with the signature, stale defaults, broken
  `:py:` cross-references or rst that won't build, comments contradicting the code.
  **Report these for the user to reconcile — never blindly rewrite either side.**

## Workflow

1. **Resolve scope** (see Arguments). State which files you'll review.
2. **Read-only sweep.** Read each file; check public API against its docstring, body
   comments against the buckets. Do not edit yet.
3. **Report**, grouped A–E, as a concise list per bucket with `file:line` refs and a
   one-line "what". Lead with highest-value, lowest-risk. Flag anything destructive.
4. **Apply only if asked** (`--fix` or explicit go-ahead):
   - Auto-apply A (after confirming large blocks) and B.
   - Do C with care, preserving NumPy/rst style; for `docs/*.rst` additions, match the
     tone of the existing pages (e.g. `docs/boundary_conditions.rst`).
   - Never touch E automatically; leave it as a checklist for the user.
   - Behaviour must not change: this is comments/docs only. Suggest the user check the
     Sphinx build (`docs/`, needs the docs requirements) if rst was touched.
   - Never commit or push — the user handles all git operations.

## Notes specific to fedoo

- Sphinx config: `docs/conf.py` — autodoc + napoleon (NumPy style) + autosummary +
  sphinx-gallery over `examples/` + pyvista plot directives. Docstring rst must build.
- Public API is what's reachable as `fd.*` / documented in `docs/*.rst` autosummaries;
  align new docstrings with the existing pages (`Mesh.rst`, `Problem.rst`,
  `WeakForm.rst`, `ConstitutiveLaw.rst`, `Assembly.rst`, `boundary_conditions.rst`).
- State conventions where users will look for them: DOF/variable ordering, Voigt order,
  sign conventions of weak forms, ramped vs step loads. If a convention is shared across
  modules, it deserves a `docs/*.rst` section that docstrings link to.
- Gallery examples double as docs *and* regression material — a new feature usually
  warrants a small `examples/` script rather than a long docstring example.
