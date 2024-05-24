"""
    ImplicitLikelihood(
        obs,
        data,
        name=nameof(obs),
    )

Represents a `Dirac` delta likelihood function which assigns full probability
mass to a single point, i.e. the predicted value of the simulator observable.
"""
ImplicitLikelihood(
    obs,
    data,
    name=nameof(obs),
) = SimulatorLikelihood(:Implicit, obs, data, nothing, name)

predictive_distribution(::SimulatorLikelihood{:Implicit}) = error("predictive distribution not defined for implicit likelihoods")
