# Builtins stub used in dictionary-related test cases.

from typing import (
    TypeVar, Generic, Iterable, Iterator, Mapping, Tuple, overload, Optional, Union, Sequence
)

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')

class object:
    def __init__(self) -> None: pass
    def __init_subclass__(cls) -> None: pass
    def __eq__(self, other: object) -> bool: pass

class type:
    __annotations__: Mapping[str, object]

class dict(Mapping[KT, VT]):
    @overload
    def __init__(self, **kwargs: VT) -> None: pass
    @overload
    def __init__(self, arg: Iterable[Tuple[KT, VT]], **kwargs: VT) -> None: pass
    def __getitem__(self, key: KT) -> VT: pass
    def __setitem__(self, k: KT, v: VT) -> None: pass
    def __iter__(self) -> Iterator[KT]: pass
    def __contains__(self, item: object) -> int: pass
    def update(self, a: Mapping[KT, VT]) -> None: pass
    @overload
    def get(self, k: KT) -> Optional[VT]: pass
    @overload
    def get(self, k: KT, default: Union[VT, T]) -> Union[VT, T]: pass
    def __len__(self) -> int: ...

class int: # for convenience
    def __add__(self, x: Union[int, complex]) -> int: pass
    def __radd__(self, x: int) -> int: pass
    def __sub__(self, x: Union[int, complex]) -> int: pass
    def __neg__(self) -> int: pass
    real: int
    imag: int

class str: pass # for keyword argument key type
class bytes: pass

class list(Sequence[T]): # needed by some test cases
    def __getitem__(self, x: int) -> T: pass
    def __iter__(self) -> Iterator[T]: pass
    def __mul__(self, x: int) -> list[T]: pass
    def __contains__(self, item: object) -> bool: pass
    def append(self, item: T) -> None: pass

class tuple(Generic[T]): pass
class function: pass
class float: pass
class complex: pass
class bool(int): pass

class ellipsis:
    __class__: object
def isinstance(x: object, t: Union[type, Tuple[type, ...]]) -> bool: pass
class BaseException: pass

def iter(__iterable: Iterable[T]) -> Iterator[T]: pass
