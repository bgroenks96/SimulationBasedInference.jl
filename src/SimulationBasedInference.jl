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
    # Extension loading;
    # We use Requires.jl instead of built-in extensions for now (v1.9.3).
    # This is due to built-in extensions having the incredibly unfortunate limitation
    # of not being loadable or exportable at precompile time... so there is no way to
    # re-export types defined within them.
    @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin
        include("../ext/SimulationBasedInferenceTuringExt/SimulationBasedInferenceTuringExt.jl")
        @reexport using .SimulationBasedInferenceTuringExt
    end
end

end
