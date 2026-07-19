"""Backward-compatibility helpers for deprecating and aliasing public API.

fedoo's stability rule toward 1.0 is *"no rename ships without its alias"*.
These helpers make that cheap and uniform:

* :func:`deprecated_alias` keeps a renamed function or method callable under its
  old name, forwarding to the new implementation and emitting a
  :class:`DeprecationWarning`.
* :func:`deprecated` is the decorator form, for a method that is itself the old
  name.
* :func:`rename_kwargs` remaps deprecated keyword-argument names, so a renamed
  constructor/method parameter keeps accepting its old spelling.

All warnings name both the old and the new symbol so users get an actionable
message.
"""

import functools
import warnings

__all__ = ["deprecated_alias", "deprecated", "rename_kwargs"]


def deprecated_alias(new, old_name=None):
    """Return a deprecated alias that forwards to ``new`` and warns.

    Keeps a renamed function or method callable under its old name::

        def compute_stiffness(self, ...):
            ...
        get_stiffness = deprecated_alias(compute_stiffness, "get_stiffness")

    Parameters
    ----------
    new : callable
        The current implementation the alias forwards to.
    old_name : str, optional
        Name reported in the warning. Defaults to ``new.__name__``; pass the old
        name explicitly so the message points at what the user actually typed.
    """

    @functools.wraps(new)
    def _alias(*args, **kwargs):
        warnings.warn(
            f"'{old_name or new.__name__}' is deprecated and will be removed in a "
            f"future release; use '{new.__name__}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new(*args, **kwargs)

    return _alias


def deprecated(new_name, since=None):
    """Decorator marking a function/method as deprecated in favour of ``new_name``."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            msg = f"'{func.__name__}' is deprecated"
            if since:
                msg += f" since fedoo {since}"
            msg += f"; use '{new_name}' instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return _wrapped

    return _decorator


def rename_kwargs(func_name, kwargs, aliases):
    """Remap deprecated keyword-argument names in place, emitting a warning.

    ``aliases`` maps ``old -> new``. Call at the top of a function whose kwargs
    were renamed::

        def __init__(self, *args, **kwargs):
            rename_kwargs(
                "Newmark", kwargs, {"StiffnessAssembly": "stiffness_assembly"}
            )

    Raises ``TypeError`` if both the old and new name are supplied.
    """
    for old, new in aliases.items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError(
                    f"{func_name}() received both '{old}' (deprecated) and '{new}'."
                )
            warnings.warn(
                f"'{old}' is a deprecated argument name for {func_name}(); "
                f"use '{new}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(old)
    return kwargs
