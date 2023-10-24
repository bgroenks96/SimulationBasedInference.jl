using SimulationBasedInference

using LogDensityProblems
using OrdinaryDiffEq
using Test

@testset "Forward problem" begin
    p = ComponentArray(α=0.1)
    odeprob = ODEProblem((u,p,t) -> -p.α*u, [1.0], (0.0,1.0), p)
    observable = BufferedObservable(:obs, state -> state.u, 0.0, 0.1:0.1:1.0, samplerate=0.01)
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    forward_sol = solve(forwardprob, Tsit5())
    @test isa(forward_sol, SimulatorForwardSolution)
    obs = retrieve(observable)
    @test size(obs) == (1,10)
    @test all(diff(obs[1,:]) .< 0.0)
end
@testset "Inference problem" begin
    ode_p = ComponentArray(α=0.1)
    odeprob = ODEProblem((u,p,t) -> -p.α*u, [1.0], (0.0,1.0), ode_p)
    observable = BufferedObservable(:obs, state -> state.u, 0.0, 0.1:0.1:1.0, samplerate=0.01)
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    prior = PriorDistribution(:α, LogNormal(0,1))
    noise_scale_prior = PriorDistribution(:σ, Exponential(1.0))
    lik = MvGaussianLikelihood(:obs, observable, noise_scale_prior)
    data = randn(10)
    inferenceprob = SimulatorInferenceProblem(forwardprob, Tsit5(), prior, lik => data)
    u = copy(inferenceprob.u0)
    @test hasproperty(u, :model) && hasproperty(u, :obs)
    u.model.α = 0.5
    u.obs.σ = 1.0
    lp = logprob(inferenceprob, u)
    # check that the logprob is equal to the logpdf of a manually constructed likelihood
    @test lp == logpdf(MvNormal(retrieve(observable)[1,:], I), data)
    # check LogDensityProblems interface
    ldpcheck = LogDensityProblems.capabilities(inferenceprob)
    @test isa(ldpcheck, LogDensityProblems.LogDensityOrder{0})
    @test LogDensityProblems.dimension(inferenceprob) == length(u)
    @test LogDensityProblems.logdensity(inferenceprob, u) == lp
end
