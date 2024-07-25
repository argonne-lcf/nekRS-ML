from __future__ import annotations

import typing as t

from .representation import Scope
from .scope_definition import MutableScope

if t.TYPE_CHECKING:
    from globus_sdk._types import ScopeCollectionType


def scopes_to_str(scopes: ScopeCollectionType) -> str:
    """
    Convert a scope collection to a space-separated string.

    :param scopes: A scope string or object, or an iterable of scope strings or objects.
    """
    return " ".join(_iter_scope_collection(scopes))


def _iter_scope_collection(obj: ScopeCollectionType) -> t.Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, (MutableScope, Scope)):
        yield str(obj)
    else:
        for item in obj:
            yield str(item)
