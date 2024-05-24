using AbstractGPs, KernelFunctions

struct GPPrior{kernelType,priorType<:AbstractSimulatorPrior} <: AbstractSimulatorPrior
    kernel::kernelType
    prior::priorType
end

Base.rand(rng::AbstractRNG, prior::GPPrior) = rand(rng, prior.prior)

logprob(prior::GPPrior, x) = logprob(prior.prior, x)

"""
    GPLikelihood(obs, data, kernelfn, prior::AbstractSimulatorPrior, name=nameof(obs))

Constructs a GP likelihood from the given observable, data, kernel function, and prior.
The kernel function should be a function that takes parameters matching those sampled from `prior`
and constructs any `Kernel` from `KernelFunctions`.
"""
GPLikelihood(obs, data, kernelfn, prior::AbstractSimulatorPrior, name=nameof(obs)) = SimulatorLikelihood(GP, obs, data, GPPrior(kernelfn, prior), name)

function predictive_distribution(lik::SimulatorLikelihood{GP}, args...)
    k = lik.prior.kernel(args...)
    x = vec(map(collect, Iterators.product(coordinates(lik.obs))))
    Σ = kernelmatrix(k, x)
    μ = vec(retrieve(lik.obs))
    return MvNormal(μ, Σ)
end
