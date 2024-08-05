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

function predictive_distribution(lik::SimulatorLikelihood{Dirac})
    y = getvalue(lik.obs)
    return Dirac(y)
end
