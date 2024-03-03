"""
    MCMC{algType,stratType,kwargType} <: SimulatorInferenceAlgorithm

Generic container type for Markov Chain Monte Carlo (MCMC) based inference algorithms.
"""
struct MCMC{algType,stratType,kwargType} <: SimulatorInferenceAlgorithm
    alg::algType # sampling algorithm
    strat::stratType # sampling strategy (when defined)
    nsamples::Int # number of posterior samples
    nchains::Int # number of chains
    kwargs::kwargType # additional kwargs for the sampler
end

function MCMC(alg, strat=MCMCSerial(); nsamples=1000, nchains=2, kwargs...)
    return MCMC(alg, strat, nsamples, nchains, (; kwargs...))
end
