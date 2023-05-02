from qompy.qube import Qircuit
from qompy.qobjs import Ket, Operator

import numpy as np


def test_fourier():
    fourier_qircuit = Qircuit(3)

    omega = np.exp(2j * np.pi / 8)
    R = Operator(np.array([[1., 0.], [0., omega]]))

    fourier_qircuit.h(0)
    fourier_qircuit.cunitary(0, 1, R)
    fourier_qircuit.cunitary(0, 2, R)
    fourier_qircuit.h(1)
    fourier_qircuit.cunitary(1, 2, R)
    fourier_qircuit.h(2)

    assert np.allclose(fourier_qircuit.evolve(Ket.basis(3))[0].data, (0.125) ** 0.5, 1e-4)

    inv_fourier_qircuit = Qircuit(3)
    R_dag = Operator(R.data.conj())

    fourier_qircuit.h(2)
    fourier_qircuit.cunitary(1, 2, R)
    fourier_qircuit.h(1)
    fourier_qircuit.cunitary(0, 2, R)
    fourier_qircuit.cunitary(0, 1, R)
    fourier_qircuit.h(0)

    transformed = fourier_qircuit.evolve(Ket.basis(3))[0]
    assert np.allclose(inv_fourier_qircuit.evolve(transformed)[0].data, Ket.basis(3).data, 1e-4)
