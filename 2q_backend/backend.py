from pkg_req import * 
from construct_H import *
from construct_H import *
from solver import *

#DynamicsBackend will (1) compute the state evolution for the system over time; (2) simulate measurement processes, compute observables, or handle measurements; (3) manage the individual components or subsystems of the full system (which can have different dimensionalities) for a multi qubit system

backend = DynamicsBackend(
    solver=solver,
    subsystem_dims=[dim, dim],  # dimensions of the subsystems (or quantum registers)
    solver_options=solver_options,  # important to supply the numerical method, precision, and optimization settings. Passed through each iteration
)