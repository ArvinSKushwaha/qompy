from qompy.qube import Qircuit
from qompy.qobjs import Ket, Operator

import numpy as np


def test_qubit():
    for i in range(3):
        qircuit = Qircuit(3)
        qircuit.q_register.h(i)

        output_state = qircuit.evolve(Ket.basis(3))[0].data
        correct_state = np.zeros((8, 1))
        correct_state[0] = 1
        correct_state[1 << (2 - i)] = 1

        correct_state /= np.sqrt(2)

        assert np.allclose(output_state, correct_state, 1e-4)


def test_cnot():
    qircuit = Qircuit(3)
    qircuit.q_register.h(0)
    qircuit.q_register.h(1)
    qircuit.q_register.cnot(0, 2)

    output_state = qircuit.evolve(Ket.basis(3))[0].data
    correct_state = np.zeros((8, 1))
    correct_state[0b000] = 1
    correct_state[0b001] = 0
    correct_state[0b010] = 1
    correct_state[0b011] = 0
    correct_state[0b100] = 0
    correct_state[0b101] = 1
    correct_state[0b110] = 0
    correct_state[0b111] = 1
    correct_state /= np.sqrt(4)

    assert np.allclose(output_state, correct_state, 1e-4)


def test_cunitary():
    X = Operator(np.array([[0., 1.], [1., 0.]]))

    qircuit = Qircuit(3)
    qircuit.q_register.h(0)
    qircuit.q_register.h(1)
    qircuit.q_register.cunitary(0, 2, X)

    output_state = qircuit.evolve(Ket.basis(3))[0].data
    correct_state = np.zeros((8, 1))
    correct_state[0b000] = 1
    correct_state[0b001] = 0
    correct_state[0b010] = 1
    correct_state[0b011] = 0
    correct_state[0b100] = 0
    correct_state[0b101] = 1
    correct_state[0b110] = 0
    correct_state[0b111] = 1
    correct_state /= np.sqrt(4)

    assert np.allclose(output_state, correct_state, 1e-4)

def qft():
    pass
