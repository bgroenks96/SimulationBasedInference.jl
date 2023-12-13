using SimulationBasedInference

using Dates
using Test

@testset "Transient" begin
    obs = SimulatorObservable(:obs)
    state = [0.0]
    SimulationBasedInference.initialize!(obs, state)
    @test obs.output.state === state
    SimulationBasedInference.observe!(obs, state)
    observed_state = SimulationBasedInference.retrieve(obs)
    @test observed_state === state
    # with observable mapping function
    state = [-1.0,2.0]
    obs = SimulatorObservable(:obs, x -> x.^2, ndims=2)
    SimulationBasedInference.initialize!(obs, state)
    SimulationBasedInference.observe!(obs, state)
    observed_state = SimulationBasedInference.retrieve(obs)
    @test observed_state == state.^2
end

@testset "TimeSampled" begin
    obsfunc(state) = state.x
    t0 = DateTime(2000,1,1)
    savepoints = t0+Day(1):Day(1):DateTime(2001,1,1)
    buffered_observable = SimulatorObservable(:testobs, obsfunc, t0, savepoints; samplerate=Hour(1))
    @test buffered_observable.output.tsave == collect(savepoints)
    @test all(diff(buffered_observable.output.tsample) .== Hour(1))
    # case 1: scalar state
    state = (x = 0.0,)
    SimulationBasedInference.initialize!(buffered_observable, state)
    @test typeof(buffered_observable.output.output) <: Vector{Float64}
    @test typeof(buffered_observable.output.buffer) <: Vector{Float64}
    # update observable at each sample point
    for t in t0:Hour(1):savepoints[end]
        state = (x = 1.0,)
        SimulationBasedInference.observe!(buffered_observable, state)
    end
    obs_result = SimulationBasedInference.retrieve(buffered_observable)
    # we save 1.0 at each step, so average should always be 1
    @test all(obs_result .≈ 1.0)
    # case 2: vector state
    buffered_observable = SimulatorObservable(:testobs, obsfunc, t0, savepoints; samplerate=Hour(1), ndims=10)
    state = (x = ones(10),)
    SimulationBasedInference.initialize!(buffered_observable, state)
    @test typeof(buffered_observable.output.output) <: Vector{Vector{Float64}}
    @test typeof(buffered_observable.output.buffer) <: Vector{Vector{Float64}}
end
