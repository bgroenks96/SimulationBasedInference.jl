GaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=Exponential(1.0)),
    name=nameof(obs)
) = SimulatorLikelihood(Normal, obs, data, prior, name)

IsotropicGaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=Exponential(1.0)),
    name=nameof(obs)
) = SimulatorLikelihood(IsoNormal, obs, data, prior, name)

DiagonalGaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=filldist(Exponential(1.0), prod(size(obs)))),
    name=nameof(obs)
) = SimulatorLikelihood(DiagNormal, obs, data, prior, name)


function predictive_distribution(lik::SimulatorLikelihood{Normal}, σ)
    μ = retrieve(lik.obs)[1]
    return Normal(μ, σ)
end

function predictive_distribution(lik::SimulatorLikelihood{<:MvNormal}, σ)
    μ = vec(retrieve(lik.obs))
    Σ = cov(lik, σ)
    return MvNormal(μ, Σ)
end

Statistics.cov(lik::SimulatorLikelihood{IsoNormal}, σ::Number) = Diagonal(σ^2*ones(prod(size(lik.data))))
Statistics.cov(lik::SimulatorLikelihood{IsoNormal}, σ::AbstractVector) = cov(lik, σ[1])
Statistics.cov(lik::SimulatorLikelihood{DiagNormal}, σ::AbstractVector) = Diagonal(σ.^2)
