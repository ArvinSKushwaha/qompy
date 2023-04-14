import qompy as qp

# Based off of this example: https://qiskit.org/textbook/ch-algorithms/deutsch-
# jozsa.html#3.-Creating-Quantum-Oracles--

const_oracle = qp.Qircuit(4)
const_oracle.x(2)

balanced_oracle = qp.Qircuit(4)
b_str = "111"

for i, c in enumerate(b_str):
    if c == "1":
        balanced_oracle.x(i)

balanced_oracle.cnot(0, 3)
balanced_oracle.cnot(1, 3)
balanced_oracle.cnot(2, 3)

for i, c in enumerate(b_str):
    if c == "1":
        balanced_oracle.x(i)

deutsch_jozsa = qp.Qircuit(4)

deutsch_jozsa.h(0)
deutsch_jozsa.h(1)
deutsch_jozsa.h(2)

deutsch_jozsa.x(3)
deutsch_jozsa.h(3)


finalize = qp.Qircuit(4, 3)

finalize.h(0)
finalize.h(1)
finalize.h(2)

finalize.measure(0, 0)
finalize.measure(1, 1)
finalize.measure(2, 2)

start = qp.Ket.basis(4)
initialized, _ = deutsch_jozsa.evolve(start)

const_oracled, _ = const_oracle.evolve(initialized)
balanced_oracled, _ = balanced_oracle.evolve(initialized)

const_result = finalize.evolve(const_oracled)
balanced_result = finalize.evolve(balanced_oracled)

print("For constant oracle")
print(const_result[0].data)
print(const_result[1])

print("For balanced oracle")
print(balanced_result[0].data)
print(balanced_result[1])

# Half-Adder
# Modular-Adder
# Quantum Phase Estimation
