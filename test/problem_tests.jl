using SimulationBasedInference

using ComponentArrays
using OrdinaryDiffEq
using Test

@testset "Forward problem interface" begin
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

@testset "Inference problem interface" begin
    p = ComponentArray(α=0.1)
    odeprob = ODEProblem((u,p,t) -> -p.α*u, [1.0], (0.0,1.0), p)
    observable = BufferedObservable(:obs, state -> state.u, 0.0, 0.1:0.1:1.0, samplerate=0.01)
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    inferenceprob = SimulatorInferenceProblem(forwardprob, prior, )
end

p = ComponentArray(α=0.1)
odeprob = ODEProblem((u,p,t) -> -p.α*u, [1.0], (0.0,1.0), p)
observable = BufferedObservable(:obs, state -> state.u, 0.0, 0.1:0.1:1.0, samplerate=0.01)
forwardprob = SimulatorForwardProblem(odeprob, observable)
