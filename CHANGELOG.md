# Changelog

All notable changes to fedoo are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and fedoo aims to follow
semantic versioning.

## [Unreleased]

### Changed

- **fedoo now requires `simcoon >= 2.0.0b1`** (previously `>= 1.14`). fedoo 1.0
  targets the simcoon 2.0 series, whose first release is the `2.0.0b1` beta.

### Migration — simcoon 2.0 `tangent_mode`

simcoon 2.0 **renumbered** the `umat()` tangent-operator enum. The mapping is:

| meaning                              | pre-2.0 | 2.0 |
| ------------------------------------ | :-----: | :-: |
| none (elastic operator)              |    –    |  0  |
| continuum tangent (default)          |    0    |  1  |
| Simo–Hughes algorithmic (consistent) |    1    |  2  |

fedoo's `Simcoon` law now defaults to `tangent_mode = 1` (continuum), preserving
the pre-2.0 numerical behavior and robustness. The algorithmic tangent stays
available as an explicit opt-in via `material.tangent_mode = 2`.

**Action required:** any code that passed **integer literals** for
`tangent_mode` must re-map them: **old `0` → `1`, old `1` → `2`**. Note that
`tangent_mode = 0` now selects *no* tangent (the elastic operator), which will
silently degrade convergence/accuracy if used unintentionally.
