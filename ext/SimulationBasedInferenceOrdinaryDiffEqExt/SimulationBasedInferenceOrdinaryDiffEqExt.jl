module SimulationBasedInferenceOrdinaryDiffEqExt

using ..OrdinaryDiffEq

using SimulationBasedInference

OrdinaryDiffEq.postamble!(solver::DiffEqSimulatorForwardSolver) = OrdinaryDiffEq.postamble!(solver.integrator)

end
