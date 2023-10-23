abstract type SimulatorInferenceAlgorithm end

function CommonSolve.solve(
    prob::SimulatorInferenceProblem,
    inferencealg::SimulatorInferenceAlgorithm,
    ode_alg,
    args...;
    kwargs...
)
    error("solve not implemented for the given algorithm: $(typeof(inferencealg))")
end

include("eks_alg.jl")
