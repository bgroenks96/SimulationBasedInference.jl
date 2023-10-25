module SimulationBasedInferenceOrdinaryDiffEqExt

using ..OrdinaryDiffEq

using SimulationBasedInference

OrdinaryDiffEq.postamble!(solver::SimulatorODEForwardSolver) = OrdinaryDiffEq.postamble!(solver.integrator)

end
