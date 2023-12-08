using Test
using Random
using SimulationBasedInference

function test_ensemble_alg_interface(solver::EnsembleSolver)
    ens = Ensembles.get_ensemble(solver.state)
    @test isa(ens, AbstractMatrix)
    obs_mean = Ensembles.get_obs_mean(solver.state)
    @test isa(obs_mean, AbstractVector)
    obs_cov = Ensembles.get_obs_cov(solver.state)
    @test isa(obs_cov, AbstractMatrix)
    @test isa(hasconverged(solver.alg, solver.state), Bool)
    rng = MersenneTwister(1234)
    initial_state = Ensembles.initialstate(solver.alg, ens, obs_mean, obs_cov; rng)
    @test isa(initial_state, Ensembles.EnsembleState)
    prev_iter = solver.state.iter
    SimulationBasedInference.step!(solver)
    @test solver.state.iter == prev_iter+1
end

include("../testcases.jl")

@testset "EKS" begin
    include("eks_tests.jl")
end

@testset "ES-MDA" begin
    include("esmda_tests.jl")
end

@testset "PBS" begin
    include("pbs_tests.jl")
end
