from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.sparse import kron, eye

from .qobjs import Operator


class Operation(ABC):
    __slots__ = ["operator", "qubes"]

    operator: Operator
    qubes: tuple[int, ...]

    def __init__(self, operator, *qubits: int) -> None:
        self.qubes = qubits
        self.operator = operator

    def argcount(self) -> int:
        return len(self.qubes)

    @abstractmethod
    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        if not state.flags.writeable:
            raise ValueError("Passed state is not writable")

        if not self._valid_state(state):
            raise ValueError("Invalid state size passed in")

    def _valid_state(self, state: npt.NDArray[np.complex128]) -> bool:
        _range = 1 << (max(self.qubes) + 1)
        state_size = len(state)
        return state_size >= _range and state_size & (state_size - 1) == 0


class Identity(Operation):
    def __init__(self, *qs: int) -> None:
        operator = Operator(np.eye(2 ** len(qs)))
        super().__init__(operator, *qs)

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)


class Hadamard(Operation):
    def __init__(self, q0: int) -> None:
        operator = Operator(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0))
        qubes = (q0,)
        super().__init__(operator, *qubes)

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)
        n_qubits = len(state).bit_length() - 1
        state[:] = (
            kron(
                eye(1 << self.qubes[0]),
                kron(self.operator.data, eye(
                    1 << (n_qubits - self.qubes[0] - 1))),
            )
            @ state[:]
        )


class PauliX(Operation):
    def __init__(self, q0: int) -> None:
        operator = Operator(np.array([[0.0, 1.0], [1.0, 0.0]]))
        qubes = (q0,)
        super().__init__(operator, *qubes)

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)
        n_qubits = len(state).bit_length() - 1
        state[:] = (
            kron(
                eye(1 << self.qubes[0]),
                kron(self.operator.data, eye(
                    1 << (n_qubits - self.qubes[0] - 1))),
            )
            @ state[:]
        )


class PauliY(Operation):
    def __init__(self, q0: int) -> None:
        operator = Operator(np.array([[0j, -1j], [1j, 0j]]))
        qubes = (q0,)
        super().__init__(operator, *qubes)

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)
        n_qubits = len(state).bit_length() - 1
        state[:] = (
            kron(
                eye(1 << self.qubes[0]),
                kron(self.operator.data, eye(
                    1 << (n_qubits - self.qubes[0] - 1))),
            )
            @ state[:]
        )


class PauliZ(Operation):
    def __init__(self, q0: int) -> None:
        operator = Operator(np.diag([1, -1]))
        qubes = (q0,)
        super().__init__(operator, *qubes)

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)
        n_qubits = len(state).bit_length() - 1
        state[:] = (
            kron(
                eye(1 << self.qubes[0]),
                kron(self.operator.data, eye(
                    1 << (n_qubits - self.qubes[0] - 1))),
            )
            @ state[:]
        )


class CNot(Operation):
    def __init__(self, q_control: int, q_target: int) -> None:
        if q_control == q_target:
            raise ValueError(
                "The qubit controlling the operation cannot also be the target"
            )

        operator = Operator(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
        qubes = (q_control, q_target)
        super().__init__(operator, *qubes)

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)

        # Reshape into form (X, 2, Y, 2, Z), where the 2 axes are the
        # ones that are considered by the operation.

        state = state[None, :, None]
        q_control, q_target = self.qubes

        q_min, q_max = min(q_control, q_target), max(q_control, q_target)
        state = state.reshape(1 << q_min, 2, 1 << (q_max - q_min - 1), 2, -1)

        if q_min == q_control:
            state[:, 1, ...] = state[:, 1, :, ::-1, ...]
        if q_max == q_control:
            state[:, :, :, 1, ...] = state[:, ::-1, :, 1, ...]


class CUnitary(Operation):
    def __init__(self, q_control: int, q_target: int, operator: Operator) -> None:
        if q_control == q_target:
            raise ValueError(
                "The qubit controlling the operation cannot also be the target"
            )

        qubes = (q_control, q_target)
        super().__init__(operator, *qubes)

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)

        # Reshape into form (X, 2, Y, 2, Z), where the 2 axes are the
        # ones that are considered by the operation.
        n_qubits = len(state).bit_length() - 1

        state = state[None, :, None]
        q_control, q_target = self.qubes

        q_min, q_max = min(q_control, q_target), max(q_control, q_target)
        state = state.reshape(1 << q_min, 2, 1 << (q_max - q_min - 1), 2, -1)

        if q_min == q_control:
            state[:, 1] = (
                kron(
                    eye(1 << (q_target - 1)),
                    kron(self.operator.data, eye(1 << (n_qubits - q_target - 1))),
                )
                @ state[:, 1].reshape(-1)
            ).reshape(1 << q_min, 1 << (q_max - q_min - 1), 2, -1)
        if q_max == q_control:
            state[:, :, :, 1, ...] = (
                kron(
                    eye(1 << q_target),
                    kron(self.operator.data, eye(
                        1 << (n_qubits - q_target - 2))),
                )
                @ state[:, :, :, 1, ...].reshape(-1)
            ).reshape(1 << q_min, 2, 1 << (q_max - q_min - 1), -1)


class Measure(Operation):
    def __init__(self, q0: int, c_register: list[bool | None], c0: int) -> None:
        operator = Operator(np.eye(2))
        qubes = (q0,)
        super().__init__(operator, *qubes)
        self.c_register = c_register
        self.c0 = c0

    def execute(self, state: npt.NDArray[np.complex128]) -> None:
        super().execute(state)
        q0, = self.qubes

        prob_zero, _ = (np.abs(state.reshape(1 << q0, 2, -1)) ** 2).\
            sum(axis=0).\
            sum(axis=1)
        if np.random.rand() < prob_zero:
            state.reshape(1 << q0, 2, -1)[:, 1, ...] = 0
            self.c_register[self.c0] = False
        else:
            state.reshape(1 << q0, 2, -1)[:, 0, ...] = 0
            self.c_register[self.c0] = True

        state[:] /= np.linalg.norm(state)
