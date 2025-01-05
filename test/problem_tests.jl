using SimulationBasedInference

using LinearAlgebra
using LogDensityProblems
using NonlinearSolve
using OrdinaryDiffEq
using Test

@testset "Forward ODEProblem" begin
    p = ComponentArray(α=0.1)
    odeprob = ODEProblem((u,p,t) -> -p.α*u, [1.0], (0.0,1.0), p)
    observable = SimulatorObservable(:u, state -> state.u, 0.0, 0.1:0.1:1.0, size(odeprob.u0), samplerate=0.01)
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    forward_sol = solve(forwardprob, Tsit5())
    @test forward_sol.sol.retcode == ReturnCode.Success
    @test isa(forward_sol, SimulatorForwardSolution)
    obs = getvalue(observable)
    @test size(obs) == (10,)
    @test all(diff(obs) .< 0.0)
end

@testset "Forward NonlinearProblem" begin
    # find zero of a polynomial
    p = ComponentArray(a=0.1, b=2.0, c=-0.5, d=1.0)
    nlprob = NonlinearProblem((u,p) -> p.a.*u.^3 .+ p.b.*u.^2 .+ p.c.*u .+ p.d, [0.0], p)
    observable = SimulatorObservable(:u, state -> state.u, size(nlprob.u0))
    forwardprob = SimulatorForwardProblem(nlprob, observable)
    forward_sol = solve(forwardprob, NewtonRaphson(), abstol=1e-6, reltol=1e-8)
    @test forward_sol.sol.retcode == ReturnCode.Success
    @test isa(forward_sol, SimulatorForwardSolution)
    obs = getvalue(observable)
    @test isa(obs, AbstractVector{Float64})
    @test round(obs[1], digits=3) == -20.271
end

@testset "Inference problem" begin
    ode_p = ComponentArray(α=0.1)
    odeprob = ODEProblem((u,p,t) -> -p[1]*u, [1.0], (0.0,1.0), ode_p)
    observable = SimulatorObservable(:obs, state -> state.u, 0.0, 0.1:0.1:1.0, size(odeprob.u0), samplerate=0.01)
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    α_prior = prior(:α, LogNormal(0,1))
    noise_scale_prior = prior(:σ, Exponential(1.0))
    data = randn(10)
    lik = SimulatorLikelihood(IsoNormal, observable, data, noise_scale_prior)
    inferenceprob = SimulatorInferenceProblem(forwardprob, Tsit5(), α_prior, lik)
    u = copy(inferenceprob.u0)
    @test hasproperty(u, :model) && hasproperty(u, :obs)
    u.model.α = 0.5
    u.obs.σ = 1.0
    lp = logprob(inferenceprob, u)
    lj = logpdf(MvNormal(getvalue(observable), I), data) + logprob(α_prior, u.model) + logprob(noise_scale_prior, u.obs)
    # check the logdensity is equal to the logjoint
    @test lp ≈ lj
    # apply bijection
    b = SBI.bijector(inferenceprob)
    x = b(u)
    ldj = SBI.logabsdetjacinv(b, x)
    # check LogDensityProblems interface
    ℓ = logdensity(inferenceprob)
    ldpcheck = LogDensityProblems.capabilities(ℓ)
    @test isa(ldpcheck, LogDensityProblems.LogDensityOrder{0})
    @test LogDensityProblems.dimension(ℓ) == length(u)
    @test LogDensityProblems.logdensity(ℓ, x) ≈ lp + ldj
end
