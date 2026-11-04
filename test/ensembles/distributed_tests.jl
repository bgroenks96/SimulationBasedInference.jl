using Distributed

const test_workers = addprocs(2; exeflags="--project=$(Base.active_project())")

@everywhere using SimulationBasedInference, OrdinaryDiffEqTsit5, FileIO, JLD2

using SimulationBasedInference: DiskStorageBackend, EnsembleForwardSolver
using Test

import Random

function distributed_test_forward_prob()
    odeprob = ODEProblem((u, p, t) -> -p[1] .* u, [1.0], (0.0, 1.0), [1.0])
    observable = SimulatorObservable(
        state -> state.u,
        size(odeprob.u0),
        name = :u,
        output = TimeSampled(0.0, 0.1:0.1:1.0, samplerate=0.01)
    )
    return SimulatorForwardProblem(odeprob, observable)
end

@testset "EnsembleDistributed: forward ensemble (in-memory)" begin
    forward_prob = distributed_test_forward_prob()
    P = rand(Random.MersenneTwister(1234), 1, 6)
    ref_sols = solve(forward_prob, Tsit5(), EnsembleThreads(); p=P)
    # default simdata
    sols = solve(forward_prob, Tsit5(), EnsembleDistributed(); p=P)
    @test length(sols) == size(P, 2)
    @test all(sol -> sol.sol.retcode == ReturnCode.Success, sols)
    @test all(get_observable.(sols, :u) .≈ get_observable.(ref_sols, :u))
    # caller-provided simdata in a shared dataset sees the results
    dataset = SimulationDataSet()
    simdata = allocate!(dataset, P)
    sols = solve(forward_prob, Tsit5(), EnsembleDistributed(); p=P, simdata)
    @test all(sol.simdata === data for (sol, data) in zip(sols, simdata))
    for i in 1:size(P, 2)
        @test getinputs(dataset, i) == P[:, i]
        @test vec(reduce(hcat, getoutputs(dataset, i).u)) ≈ get_observable(ref_sols[i], :u)
    end
end

@testset "EnsembleDistributed: forward ensemble (JLD2)" begin
    forward_prob = distributed_test_forward_prob()
    P = rand(Random.MersenneTwister(1234), 1, 6)
    ref_sols = solve(forward_prob, Tsit5(), EnsembleThreads(); p=P)
    mktempdir() do dir
        dataset = SimulationDataSet(DiskStorageBackend(format"JLD2", dir))
        simdata = allocate!(dataset, P)
        sols = solve(forward_prob, Tsit5(), EnsembleDistributed(); p=P, simdata)
        @test all(sol -> sol.sol.retcode == ReturnCode.Success, sols)
        @test all(get_observable.(sols, :u) .≈ get_observable.(ref_sols, :u))
        for i in 1:size(P, 2)
            @test vec(reduce(hcat, getoutputs(dataset, i).u)) ≈ get_observable(ref_sols[i], :u)
        end
    end
end

@testset "EnsembleDistributed: remote solver state persists across step!" begin
    forward_prob = distributed_test_forward_prob()
    P = rand(Random.MersenneTwister(1234), 1, 4)
    ref_sols = solve(forward_prob, Tsit5(), EnsembleThreads(); p=P)
    enssolver = init(remake(forward_prob; p=P), Tsit5(), EnsembleDistributed())
    @test isa(enssolver, EnsembleForwardSolver{EnsembleDistributed})
    # solvers are distributed over the workers
    @test sort(unique(map(ref -> ref.where, enssolver.solvers))) == sort(test_workers)
    # stepping advances the remote solvers; solve! then continues from where they left off
    step!(enssolver)
    step!(enssolver)
    sols = solve!(enssolver)
    @test all(get_observable.(sols, :u) .≈ get_observable.(ref_sols, :u))
    @test all(data -> length(getoutputs(data).u) == 10, enssolver.simdata)
end

@testset "EnsembleDistributed: EKS linear ODE inversion" begin
    rng = Random.MersenneTwister(1234)
    α_true = 0.2
    inference_prob = linear_ode(α_true; rng)
    ensemble_size = 32
    eks_sol = solve(inference_prob, EKS(), EnsembleDistributed(); ensemble_size, verbose=false, rng)
    posterior_mean = mean(get_transformed_ensemble(eks_sol), dims=2)
    @test abs(posterior_mean[1] - α_true) < 0.01
    # forward outputs from every iteration are visible in the inference storage
    storage = eks_sol.storage
    @test length(storage) == ensemble_size * iterations(storage)
    @test all(i -> length(getoutputs(storage, i).obs) == 10, 1:length(storage))
end

rmprocs(test_workers)
