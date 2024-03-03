module SimulationBasedInferenceDynamicHMCExt

using SimulationBasedInference

using ComponentArrays
using ForwardDiff
using LogDensityProblemsAD
using MCMCChains
using Statistics

import CommonSolve
import DynamicHMC
import Random

mutable struct DynamicHMCSolver{algType<:MCMC,probType,statsType,QType}
    sol::SimulatorInferenceSolution{algType,probType}
    steps::DynamicHMC.MCMCSteps
    stats::statsType
    Q::QType
end

include("hmc.jl")

end