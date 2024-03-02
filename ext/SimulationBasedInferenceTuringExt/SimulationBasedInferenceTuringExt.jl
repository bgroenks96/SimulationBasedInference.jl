module SimulationBasedInferenceTuringExt

using SimulationBasedInference

using Turing
using Turing.AbstractMCMC

# common dependencies
using CommonSolve
using ComponentArrays
using DiffEqBase
using Random
using StatsBase

include("utils.jl")

export TuringPrior
include("turing_prior.jl")

export TuringMCMC, joint_model, likelihood_model
include("turing_models.jl")

end
