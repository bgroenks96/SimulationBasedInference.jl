"""
    DiracLikelihood(
        obs,
        data,
        name=nameof(obs),
    )

Represents a `Dirac` delta likelihood function which assigns full probability
mass to a single point, i.e. the predicted value of the simulator observable.
"""
DiracLikelihood(
    obs,
    data,
    name=nameof(obs),
) = SimulatorLikelihood(Dirac, obs, data, nothing, name)

const ImplicitLikelihood = DiracLikelihood

function predictive_distribution(lik::SimulatorLikelihood{Dirac})
    y = retrieve(lik.obs)[1]
    return Dirac(y)
end
