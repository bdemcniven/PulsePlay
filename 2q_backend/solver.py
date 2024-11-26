from pkg_req import * 
from construct_H import *

#Solver is a qiskit class for simulating either Hamiltonian or Lindblad dynamics
#See here: https://qiskit-community.github.io/qiskit-dynamics/stubs/qiskit_dynamics.solvers.Solver.html#qiskit_dynamics.solvers.Solver
# rotating_frame defines the frame operator, which is a set of vectors in a Hilbert space providing a way to decompose vectors. Unlike a basis, the vectors in a frame may not be linearly independent and they can be overcomplete. The frame operator provides a way to measure the extent of this overcompleteness and how vectors in the space can be reconstructed from the frame elements
# rotating_frame therefore transforms the Hamiltonian to make the problem more tractable (i.e., by removing fast oscillations, time dependence, etc.)

# build solver
dt = 1 / 4.5e9
solver = Solver(
    static_hamiltonian=static_ham_full, #full hamiltonian
    hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1], #interaction operators; the driving operators each repeated twice
    rotating_frame=static_ham_full, #Defining reference frame that H is evaluated in. Typically a good idea to apply this when fast oscillations are present as it will remove negligible high frequency oscillatory components. Typically unitary transformation used to focus on low freq dynamics
    hamiltonian_channels=["d0", "d1", "u0", "u1"], #channels used to apply operators; di=driving channels, ui=unitary evolution channels
    channel_carrier_freqs={"d0": v0, "d1": v1, "u0": v1, "u1": v0}, #frequencies applied to each driving and unitary operations
    dt=dt,
    array_library="jax", #numerical backend; jax is good for automatic differentiation, GPU acceleration, and typcially used where gradients and optimization are important. Here could also apply other things like tensorflow
)
solver_options = {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax": dt}
