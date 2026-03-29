module SimulationBasedInferenceTuringExt

using SimulationBasedInference

using AbstractMCMC
using Turing
using Turing.DynamicPPL

# common dependencies
using CommonSolve
using ComponentArrays
using SciMLBase
using Random
using StatsBase

export TuringSimulatorPrior
include("turing_prior.jl")

end
