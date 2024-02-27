module SimulationBasedInferenceTuringExt

# weak dependency
using ..Turing
using ..Turing.AbstractMCMC

using SimulationBasedInference

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
