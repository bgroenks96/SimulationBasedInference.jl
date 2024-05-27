using SimulationBasedInference

using Dates
using Test

@testset "Transient" begin
    state = [0.0]
    obs = SimulatorObservable(:obs, identity, size(state))
    SimulationBasedInference.initialize!(obs, state)
    @test obs.output.state == state
    SimulationBasedInference.observe!(obs, state)
    observed_state = SimulationBasedInference.getvalue(obs)
    @test observed_state == state
    # with observable mapping function
    state = [-1.0,2.0]
    obs = SimulatorObservable(:obs, x -> x.^2, size(state))
    SimulationBasedInference.initialize!(obs, state)
    SimulationBasedInference.observe!(obs, state)
    observed_state = SimulationBasedInference.getvalue(obs)
    @test observed_state == state.^2
end

@testset "TimeSampled" begin
    obsfunc(state) = state.x
    t0 = DateTime(2000,1,1)
    savepoints = t0+Day(1):Day(1):DateTime(2001,1,1)
    # case 1: scalar state
    state = (x = 0.0,)
    buffered_observable = SimulatorObservable(:testobs, obsfunc, t0, savepoints, size(state.x); samplerate=Hour(1))
    @test buffered_observable.output.tsave == collect(savepoints)
    @test all(diff(buffered_observable.output.tsample) .== Hour(1))
    SimulationBasedInference.initialize!(buffered_observable, state)
    @test typeof(buffered_observable.output.output) <: Vector
    @test typeof(buffered_observable.output.buffer) <: Vector
    # update observable at each sample point
    for t in t0:Hour(1):savepoints[end]
        state = (x = 1.0,)
        SimulationBasedInference.observe!(buffered_observable, state)
    end
    obs_result = SimulationBasedInference.getvalue(buffered_observable)
    # we save 1.0 at each step, so average should always be 1
    @test all(obs_result .≈ 1.0)
    # case 2: vector state
    state = (x = ones(10),)
    buffered_observable = SimulatorObservable(:testobs, obsfunc, t0, savepoints, size(state.x); samplerate=Hour(1))
    SimulationBasedInference.initialize!(buffered_observable, state)
    @test typeof(buffered_observable.output.output) <: Vector
    @test typeof(buffered_observable.output.buffer) <: Vector
end
