module SimulationBasedInferenceDynamicHMCExt

using SimulationBasedInference

import CommonSolve
import DynamicHMC

function SimulationBasedInference.MCMC(
    alg::DynamicHMC.NUTS;
    kwargs...
)
    return MCMC(alg, nothing; kwargs...)
end

function CommonSolve.solve(prob::SimulatorInferenceProblem, mcmc::MCMC{<:DynamicHMC.NUTS}, ode_alg; kwargs...)
    
end

end