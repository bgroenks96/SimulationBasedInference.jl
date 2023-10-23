module SimulationBasedInference

# Utility
using Dates
using Reexport
using Requires

# SciML/DiffEq
using CommonSolve
using DiffEqBase, SciMLBase

# Stats
using Bijectors
@reexport using Distributions
using MCMCChains
using Random
using StatsBase
using Statistics

export SimulatorObservable, BufferedObservable
export samplepoints, observe!, retrieve
include("observables.jl")

export ParameterMapping
include("param_map.jl")

export AbstractPrior, PriorDistributions
include("priors.jl")

export MvGaussianLikelihood, IsotropicGaussianLikelihood, DiagonalGaussianLikelihood
include("likelihoods.jl")

export SimulatorForwardProblem, SimulatorForwardSolution, SimulatorInferenceProblem, SimulatorInferenceSolution
include("problems.jl")

include("forward_solve.jl")

export SimulatorInferenceAlgorithm, EKS, fitekp!
include("inference/inference.jl")

function __init__()
    # Extension loading:
    # We use Requires.jl to allow for re-exporting of types from the extension modules.
    # This is a crude hack to get around current (v1.9) limitations of Julia's extension feature.
    @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin
        turing_ext_module = Base.get_extension(@__MODULE__, :SimulationBasedInferenceTuringExt)
        @reexport using .turing_ext_module
    end
end

end
