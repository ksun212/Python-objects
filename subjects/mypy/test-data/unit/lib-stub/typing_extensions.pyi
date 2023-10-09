import typing
from typing import Any, Mapping, Iterator, NoReturn as NoReturn, Dict, Type
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import NewType as NewType, overload as overload

import sys

_T = typing.TypeVar('_T')

class _SpecialForm:
    def __getitem__(self, typeargs: Any) -> Any:
        pass

    def __call__(self, arg: Any) -> Any:
        pass

NamedTuple = 0
Protocol: _SpecialForm = ...
def runtime_checkable(x: _T) -> _T: pass
runtime = runtime_checkable

Final: _SpecialForm = ...
def final(x: _T) -> _T: pass

Literal: _SpecialForm = ...

Annotated: _SpecialForm = ...

TypeVar: _SpecialForm

ParamSpec: _SpecialForm
Concatenate: _SpecialForm

TypeAlias: _SpecialForm

TypeGuard: _SpecialForm
Never: _SpecialForm

TypeVarTuple: _SpecialForm
Unpack: _SpecialForm

# Fallback type for all typed dicts (does not exist at runtime).
class _TypedDict(Mapping[str, object]):
    # Needed to make this class non-abstract. It is explicitly declared abstract in
    # typeshed, but we don't want to import abc here, as it would slow down the tests.
    def __iter__(self) -> Iterator[str]: ...
    def copy(self: _T) -> _T: ...
    # Using NoReturn so that only calls using the plugin hook can go through.
    def setdefault(self, k: NoReturn, default: object) -> object: ...
    # Mypy expects that 'default' has a type variable type.
    def pop(self, k: NoReturn, default: _T = ...) -> object: ...
    def update(self: _T, __m: _T) -> None: ...
    if sys.version_info < (3, 0):
        def has_key(self, k: str) -> bool: ...
    def __delitem__(self, k: NoReturn) -> None: ...

def TypedDict(typename: str, fields: Dict[str, Type[_T]], *, total: Any = ...) -> Type[dict]: ...

def reveal_type(__obj: T) -> T: pass

def dataclass_transform(
    *,
    eq_default: bool = ...,
    order_default: bool = ...,
    kw_only_default: bool = ...,
    field_specifiers: tuple[type[Any] | Callable[..., Any], ...] = ...,
    **kwargs: Any,
) -> Callable[[T], T]: ...

_FutureFeatureFixture = 0
