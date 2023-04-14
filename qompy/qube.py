from qompy.qobjs import Ket, Operator
from collections.abc import Sequence

import numpy as np
from qompy.gates import (
    CNot,
    CUnitary,
    Hadamard,
    Identity,
    Operation,
    PauliX,
    PauliY,
    PauliZ,
    Measure
)


class Qube:
    """A computational equivalent of a qubit. Registers operations performed
    on it."""

    __slots__ = ["id", "operation", "parent"]

    id: np.uint64
    operation: Operation | None
    # parent: "Qube" | None    FIXME: Typing fails because python stoopid

    def __init__(self, operation: None | Operation = None) -> None:
        self.id = np.uint64(np.random.randint(0, np.iinfo(np.int64).max))
        self.operation = operation


class QuantumRegister:
    __slots__ = ["__qubes", "__ops"]

    __qubes: int
    __ops: list[Operation]

    def __init__(self, init_size: int = 0) -> None:
        self.__qubes = init_size
        self.__ops = []

    def __len__(self) -> int:
        return self.__qubes

    def ops(self) -> Sequence[Operation]:
        return self.__ops

    def add_qubit(self):
        self.__qubes += 1

    def measure(self, qube: int, c_register: list[bool | None], cube: int):
        self.__ops.append(Measure(qube, c_register, cube))

    def i(self, qube: int):
        """This operation is effectively the same as doing no operation
        at all."""
        if qube < 0 or qube >= self.__qubes:
            raise IndexError(f"Could not apply operation `I` to qube {qube}")

        self.__ops.append(Identity(qube))

    def h(self, qube: int):
        """Performs the Hadamard operation on the qubit at the given index."""
        if qube < 0 or qube >= self.__qubes:
            raise IndexError(f"Could not apply operation `H` to qube {qube}")

        op = Hadamard(qube)
        self.__ops.append(op)

    def x(self, qube: int):
        """Performs the Pauli X operation on the qubit at the given index."""
        if qube < 0 or qube >= self.__qubes:
            raise IndexError(f"Could not apply operation `X` to qube {qube}")

        op = PauliX(qube)
        self.__ops.append(op)

    def y(self, qube: int):
        """Performs the Pauli Y operation on the qubit at the given index."""
        if qube < 0 or qube >= self.__qubes:
            raise IndexError(f"Could not apply operation `Y` to qube {qube}")

        op = PauliY(qube)
        self.__ops.append(op)

    def z(self, qube: int):
        """Performs the Pauli Z operation on the qubit at the given index."""
        if qube < 0 or qube >= self.__qubes:
            raise IndexError(f"Could not apply operation `Z` to qube {qube}")

        op = PauliZ(qube)
        self.__ops.append(op)

    def cnot(self, control_qube: int, target_qube: int):
        """Performs the Controlled-Not operation on the qubit at the given index."""
        if control_qube < 0 or control_qube >= self.__qubes:
            raise IndexError(
                f"Could not apply operation `CX` to qubes \
            {control_qube}, {target_qube}"
            )

        if target_qube < 0 or target_qube >= self.__qubes:
            raise IndexError(
                f"Could not apply operation `CX` to qubes \
            {control_qube}, {target_qube}"
            )

        op = CNot(control_qube, target_qube)
        self.__ops.append(op)

    def cunitary(self, control_qube: int, target_qube: int, unitary: Operator):
        """Performs the Controlled-Unitary operation on the qubit at the given index."""
        if control_qube < 0 or control_qube >= self.__qubes:
            raise IndexError(
                f"Could not apply operation `CU` to qubes \
            {control_qube}, {target_qube}"
            )

        if target_qube < 0 or target_qube >= self.__qubes:
            raise IndexError(
                f"Could not apply operation `CU` to qubes \
            {control_qube}, {target_qube}"
            )

        op = CUnitary(control_qube, target_qube, unitary)
        self.__ops.append(op)


class Qircuit:
    __slots__ = ["q_register", "c_register"]

    q_register: QuantumRegister
    c_register: list[bool | None]

    def __init__(self, q_reg_size: int = 1, c_reg_size: int = 0) -> None:
        self.q_register = QuantumRegister(q_reg_size)
        self.c_register = [None for _ in range(c_reg_size)]

    def add_qubit(self):
        self.q_register.add_qubit()

    def evolve(self, state: Ket) -> tuple[Ket, list[bool | None]]:
        n_qubits = len(self.q_register)
        result = Ket(np.copy(state.data), writeable=True)

        if len(state.data) != (1 << n_qubits):
            raise ValueError(
                f"Expected a state with {1 << n_qubits} dimensions \
                but found a state with {len(state.data)} dimensions"
            )

        for operation in self.q_register.ops():
            operation.execute(result.data)

        return (result, self.c_register[:])

    def measure(self, qube: int, cube: int):
        self.q_register.measure(qube, self.c_register, cube)

    def i(self, qube: int):
        self.q_register.i(qube)

    def h(self, qube: int):
        self.q_register.h(qube)

    def x(self, qube: int):
        self.q_register.x(qube)

    def y(self, qube: int):
        self.q_register.y(qube)

    def z(self, qube: int):
        self.q_register.z(qube)

    def cnot(self, control_qube: int, target_qube: int):
        self.q_register.cnot(control_qube, target_qube)

    def cunitary(self, control_qube: int, target_qube: int, unitary: Operator):
        self.q_register.cunitary(control_qube, target_qube, unitary)
