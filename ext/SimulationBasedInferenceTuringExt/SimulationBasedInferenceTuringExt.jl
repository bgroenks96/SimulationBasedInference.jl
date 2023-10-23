module SimulationBasedInferenceTuringExt

using SimulationBasedInference

using CommonSolve
using DiffEqBase
using Random
using StatsBase
using Turing

include("utils.jl")

export TuringPrior
include("turing_prior.jl")

export TuringMCMC, joint_model, likelihood_model
include("turing_models.jl")

include("gaussian_approx.jl")

end
