using Bijectors
using ComponentArrays
using Distributions
using LinearAlgebra
using SimulationBasedInference
using Random

function evensen_scalar_nonlinear(
    x_true=1.0,
    b_true=0.2;
    n_obs=100,
    σ_y=0.1,
    x_prior=Normal(0,1),
    b_prior=autoprior(0.1, 0.5, lower=0, upper=Inf),
    σ_prior=Exponential(σ_y),
    rng=Random.default_rng(),
)
    function g(θ)
        x = θ[1]
        b = θ[2]
        return x*(1 + b*x^2)
    end
    θ_true = [x_true, b_true]
    yt = g(θ_true)
    y_obs = yt .+ σ_y*randn(rng, n_obs)
    θ_prior = prior(x=x_prior, b=b_prior)
    y_pred = SimulatorObservable(:y, state -> repeat(collect(state.u), n_obs), size(y_obs))
    forward_prob = SimulatorForwardProblem(g, θ_true, y_pred)
    lik = SimulatorLikelihood(IsoNormal, y_pred, y_obs, prior(:σ, σ_prior))
    inference_prob = SimulatorInferenceProblem(forward_prob, nothing, θ_prior, lik)
    return inference_prob
end

function linear_ode(
    α_true=0.2;
    n_obs=10,
    σ_y=0.01,
    α_prior=Beta(1,1),
    σ_prior=Exponential(σ_y),
    ode_solver=Tsit5(),
    rng=Random.default_rng(),
)
    # set up linear ODE problem with one parameter
    ode_p = ComponentArray(α=α_true)
    odeprob = ODEProblem((u,p,t) -> -p[1]*u, [1.0], (0.0,1.0), ode_p)
    # observable extracts state from integrator
    observable = SimulatorObservable(:obs, integrator -> integrator.u, 0.0, 0.1:0.1:1.0, size(odeprob.u0), samplerate=0.01)
    # specify forward problem
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    # generate "true" solution
    forward_sol = solve(forwardprob, Tsit5())
    @assert forward_sol.sol.retcode == ReturnCode.Default
    true_obs = retrieve(observable)
    # specify priors
    α_prior = prior(α=α_prior)
    noise_scale = σ_y
    noise_scale_prior = prior(:σ, σ_prior)
    # create noisy data
    noisy_obs = true_obs .+ noise_scale*randn(rng, n_obs)
    # simple Gaussian likelihood; note that we're cheating a bit here since we know the noise level a priori
    lik = SimulatorLikelihood(IsoNormal, observable, noisy_obs, noise_scale_prior)
    # inference problem
    inference_prob = SimulatorInferenceProblem(forwardprob, ode_solver, α_prior, lik)
    return inference_prob
end
