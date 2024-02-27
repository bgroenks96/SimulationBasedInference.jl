using Test
using Random
using SimulationBasedInference

function test_ensemble_alg_interface(solver::EnsembleSolver)
    prob = solver.sol.prob
    ens = SimulationBasedInference.get_ensemble(solver.state)
    @test isa(ens, AbstractMatrix)
    obs_mean = SimulationBasedInference.get_obs_mean(solver.state)
    @test isa(obs_mean, AbstractVector)
    obs_cov = SimulationBasedInference.get_obs_cov(solver.state)
    @test isa(obs_cov, AbstractMatrix)
    @test isa(SimulationBasedInference.isiterative(solver.alg), Bool)
    if SimulationBasedInference.isiterative(solver.alg)
        @test isa(SimulationBasedInference.hasconverged(solver.alg, solver.state), Bool)
    end
    rng = MersenneTwister(1234)
    initial_state = SimulationBasedInference.initialstate(solver.alg, prob.prior.model, ens, obs_mean, obs_cov; rng)
    @test isa(initial_state, SimulationBasedInference.EnsembleState)
    prev_iter = solver.state.iter
    SimulationBasedInference.step!(solver)
    @test solver.state.iter == prev_iter+1
end

include("../testcases.jl")

@testset "EnIS" begin
    include("pbs_tests.jl")
end

@testset "ES-MDA" begin
    include("esmda_tests.jl")
end

@testset "EKS" begin
    include("eks_tests.jl")
end
