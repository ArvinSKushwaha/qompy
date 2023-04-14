from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class QobjType(Enum):
    """The QobjType object is an enumeration of possible QObj types."""

    KET = auto()
    BRA = auto()
    OP = auto()


class Qobj(ABC):
    """The Qobj abstract object is a superclass of all Bras, Kets, and Ops. It
    is primarily intended for use in static typing and general complex
    operations."""

    __slots__ = ["qobj_type", "data"]

    qobj_type: QobjType
    data: npt.NDArray[np.complex128]

    def shape(self) -> tuple[int, ...] | None:
        self.data.shape

    @property
    def dag(self) -> "Qobj":
        match self.qobj_type:
            case QobjType.KET:
                return Bra(self.data.conj().transpose())
            case QobjType.BRA:
                return Ket(self.data.conj().transpose())
            case QobjType.OP:
                return Operator(self.data.conj().transpose())

    @staticmethod
    def validate_state(data: npt.NDArray, qobj_type: QobjType) -> bool:
        data = data.astype(np.complex128)
        shape = data.shape
        if shape is None or data.ndim != 2:
            return False

        match qobj_type:
            case QobjType.KET:
                return shape[1] == 1
            case QobjType.BRA:
                return shape[0] == 1
            case QobjType.OP:
                return True

    @abstractmethod
    def tensorprod(self, rhs: 'Qobj') -> 'Qobj':
        ...


class Ket(Qobj):
    def __init__(
        self, data: npt.NDArray | None = None, writeable: bool = False
    ) -> None:
        super().__init__()
        self.qobj_type = QobjType.KET

        if data is not None and self.validate_state(data, self.qobj_type):
            self.data = data
            data = data.astype(np.complex128)
        else:
            self.data = np.zeros((1, 1), dtype=np.complex128)

        self.data.flags.writeable = writeable

    def tensorprod(self, rhs: 'Ket') -> 'Ket':
        return Ket(np.tensordot(self.data, rhs.data, axes=0))

    @staticmethod
    def basis(n_qubits: int) -> 'Ket':
        state = np.zeros((1 << n_qubits, 1), dtype=np.complex128)
        state[0] = 1.

        return Ket(state)


class Bra(Qobj):
    def __init__(
        self, data: npt.NDArray | None = None, writeable: bool = False
    ) -> None:
        super().__init__()
        self.qobj_type = QobjType.BRA

        if data is not None and self.validate_state(data, self.qobj_type):
            data = data.astype(np.complex128)
            self.data = data
        else:
            self.data = np.zeros((1, 1), dtype=np.complex128)

        self.data.flags.writeable = writeable

    def tensorprod(self, rhs: 'Bra') -> 'Bra':
        return Bra(np.tensordot(self.data, rhs.data, axes=0))


class Operator(Qobj):
    def __init__(
        self, data: npt.NDArray | None = None, writeable: bool = False
    ) -> None:
        super().__init__()
        self.qobj_type = QobjType.OP
        if data is not None and self.validate_state(data, self.qobj_type):
            data = data.astype(np.complex128)
            self.data = data
        else:
            self.data = np.zeros((1, 1), dtype=np.complex128)

        self.data.flags.writeable = writeable

    def tensorprod(self, rhs: 'Operator') -> 'Operator':
        return Operator(np.tensordot(self.data, rhs.data, axes=0))
