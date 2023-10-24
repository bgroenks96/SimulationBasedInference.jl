using SimulationBasedInference

using OrdinaryDiffEq
using Test

@testset "EKS" begin
    ode_p = ComponentArray(α=0.1)
    odeprob = ODEProblem((u,p,t) -> -p.α*u, [1.0], (0.0,1.0), ode_p)
    observable = BufferedObservable(:obs, state -> state.u, 0.0, 0.1:0.1:1.0, samplerate=0.01)
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    forward_sol = solve(forwardprob, Tsit5())
    prior = PriorDistribution(:α, LogNormal(0,1))
    lik = MvGaussianLikelihood(:obs, observable)
    inferenceprob = SimulatorInferenceProblem(forwardprob, Tsit5(), prior, lik => data)
end