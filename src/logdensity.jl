"""
Internal wrapper type for `LogDensityProblems` interface.
"""
struct LogDensityProblem{funcType}
    f::funcType
    ndim::Int
end

"""
    LogDensityProblems.logdensity(
        inference_prob::SimulatorInferenceProblem;
        storage::SimulationData,
        kwargs...
    )

Constructs an internal `LogDensityProblem` wrapper type that satisfies
the `LogDensityProblems` interface.
"""
function LogDensityProblems.logdensity(
    inference_prob::SimulatorInferenceProblem;
    storage=SimulationArrayStorage(),
    kwargs...
)
    f = logdensityfunc(inference_prob, storage; kwargs...)
    return LogDensityProblem(f, length(inference_prob.u0))
end

LogDensityProblems.logdensity(ldp::LogDensityProblem, x) = ldp.f(x)

LogDensityProblems.capabilities(::Type{<:LogDensityProblem}) = LogDensityProblems.LogDensityOrder{0}()

LogDensityProblems.dimension(ldp::LogDensityProblem) = ldp.ndim
