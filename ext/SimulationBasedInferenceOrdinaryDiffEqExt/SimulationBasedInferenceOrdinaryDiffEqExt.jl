module SimulationBasedInferenceOrdinaryDiffEqExt

using SimulationBasedInference

using ..OrdinaryDiffEq

OrdinaryDiffEq.postamble!(integrator::SimulatorForwardDEIntegrator) = OrdinaryDiffEq.postamble!(integrator.integrator)

end
