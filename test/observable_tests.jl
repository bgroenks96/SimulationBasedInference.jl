using SimulationBasedInference

using Dates
using Test

@testset "BufferedObservable" begin
    samplefunc(state) = state.x
    t0 = DateTime(2000,1,1)
    savepoints = t0+Day(1):Day(1):DateTime(2001,1,1)
    buffered_observable = BufferedObservable(:testobs, samplefunc, t0, savepoints; samplerate=Hour(1))
    @test buffered_observable.tsave === savepoints
    @test all(diff(buffered_observable.tsample) .== Hour(1))
    # case 1: scalar state
    state = (x = 0.0,)
    SimulationBasedInference.init!(buffered_observable, state)
    @test typeof(buffered_observable.cache.output) <: Vector{Float64}
    @test typeof(buffered_observable.cache.buffer) <: Vector{Float64}
    # update observable at each sample point
    for t in t0:Hour(1):savepoints[end]
        state = (x = 1.0,)
        SimulationBasedInference.update!(buffered_observable, state)
    end
    obs_result = SimulationBasedInference.retrieve(buffered_observable)
    # we save 1.0 at each step, so average should always be 1
    @test all(obs_result .≈ 1.0)
    # case 2: vector state
    buffered_observable = BufferedObservable(:testobs, samplefunc, t0, savepoints; samplerate=Hour(1), ndims=10)
    state = (x = ones(10),)
    SimulationBasedInference.init!(buffered_observable, state)
    @test typeof(buffered_observable.cache.output) <: Vector{Vector{Float64}}
    @test typeof(buffered_observable.cache.buffer) <: Vector{Vector{Float64}}
end
