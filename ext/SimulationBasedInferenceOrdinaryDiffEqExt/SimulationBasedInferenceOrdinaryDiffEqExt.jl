module SimulationBasedInferenceOrdinaryDiffEqExt

using ..OrdinaryDiffEq

using SimulationBasedInference

OrdinaryDiffEq.postamble!(integrator::SimulatorForwardDEIntegrator) = OrdinaryDiffEq.postamble!(integrator.integrator)

end
