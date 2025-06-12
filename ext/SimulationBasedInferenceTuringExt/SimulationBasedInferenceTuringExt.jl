module SimulationBasedInferenceTuringExt

using SimulationBasedInference

using AbstractMCMC
using Turing
using Turing.DynamicPPL

# common dependencies
using CommonSolve
using ComponentArrays
using DiffEqBase
using Random
using StatsBase

export TuringSimulatorPrior
include("turing_prior.jl")

export TuringMCMC, joint_model, likelihood_model
include("turing_models.jl")

end
