"""
    MCMC{algType,stratType,kwargType} <: SimulatorInferenceAlgorithm

Generic container type for Markov Chain Monte Carlo (MCMC) based inference algorithms.
"""
struct MCMC{algType,stratType,kwargType} <: SimulatorInferenceAlgorithm
    alg::algType # sampling algorithm
    strat::stratType # sampling strategy (when defined)
    kwargs::kwargType # additional kwargs for the sampler
end

"""
    MCMC(alg, strat; kwargs...)

Generic consturctor for MCMC algorithms with implementation-specific type `alg`
and "strategy" `strat`.
"""
function MCMC(alg, strat=MCMCSerial(); kwargs...)
    return MCMC(alg, strat, (; kwargs...))
end
