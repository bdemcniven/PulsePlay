import pkg_req


#Hamiltonian parameters:
#dim= specifying the number of energy levels
#v_i= qubit frequencies
#anham_i=i^th Anharmonicity. This is the coefficient that introduces an anharmonic correction to the Hamiltonian, which typically arises in systems where the energy levels are not equally spaced (such as a transmon or other qubits with anharmonic potentials)
#r_i= Rabi strengths (frequency where the probability amplitudes of two different energy levels fluctuate in an oscillating EM field)
#J=coupling strength
#a/adag=lowering/raising operators
#N_i=Number operator for i^th qubit

##############################
#parameter definitions

dim = 3
v0 = 4.86e9
anharm0 = -0.32e9
r0 = 0.22e9
v1 = 4.97e9
anharm1 = -0.32e9
r1 = 0.26e9
J = 0.002e9

##############################
#operator definitions

a = np.diag(np.sqrt(np.arange(1, dim)), 1)
adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
N = np.diag(np.arange(dim))

ident = np.eye(dim, dtype=complex)
full_ident = np.eye(dim**2, dtype=complex)

N0 = np.kron(ident, N)
N1 = np.kron(N, ident)

a0 = np.kron(ident, a)
a1 = np.kron(a, ident)

a0dag = np.kron(ident, adag)
a1dag = np.kron(adag, ident)

##############################
#Hamiltonian construction

static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)


# 2 * np.pi * v0 * N0 --> linear term
# np.pi * anharm0 * N0 * (N0 - full_ident) --> Anharmonic correction
# Note that full_ident operator is used to shif/adjust static_ham terms. In this case, ensures the term only applies to N1 \greq 1

static_ham_full = (
    static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))
)

# 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag)) --> Interaction term between 2 qubits.

##############################
#Driving operator, which is the external driving field/perturbation which manipulates the state of the transmon qubit

drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)