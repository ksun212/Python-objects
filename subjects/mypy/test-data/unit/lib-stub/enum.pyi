from typing import Any, TypeVar, Union, Type, Sized, Iterator

_T = TypeVar('_T')

class EnumMeta(type, Sized):
    def __len__(self) -> int: pass  # to make it non-abstract
    def __iter__(self: Type[_T]) -> Iterator[_T]: pass
    def __reversed__(self: Type[_T]) -> Iterator[_T]: pass
    def __getitem__(self: Type[_T], name: str) -> _T: pass

class Enum(metaclass=EnumMeta):
    def __new__(cls: Type[_T], value: object) -> _T: pass
    def __repr__(self) -> str: pass
    def __str__(self) -> str: pass
    def __format__(self, format_spec: str) -> str: pass
    def __hash__(self) -> Any: pass
    def __reduce_ex__(self, proto: Any) -> Any: pass

    name: str
    value: Any
    _name_: str
    _value_: Any

    # In reality, _generate_next_value_ is python3.6 only and has a different signature.
    # However, this should be quick and doesn't require additional stubs (e.g. `staticmethod`)
    def _generate_next_value_(self) -> Any: pass

class IntEnum(int, Enum):
    value: int
    def __new__(cls: Type[_T], value: Union[int, _T]) -> _T: ...

def unique(enumeration: _T) -> _T: pass

# In reality Flag and IntFlag are 3.6 only

class Flag(Enum):
    def __or__(self: _T, other: Union[int, _T]) -> _T: pass


class IntFlag(int, Flag):
    def __and__(self: _T, other: Union[int, _T]) -> _T: pass


class auto(IntFlag):
    value: Any


# It is python-3.11+ only:
class StrEnum(str, Enum):
    def __new__(cls: Type[_T], value: str | _T) -> _T: ...
