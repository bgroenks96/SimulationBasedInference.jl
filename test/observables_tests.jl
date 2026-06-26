using SimulationBasedInference

using Dates
using Test

@testset "Transient" begin
    data = SimulationData()
    state = [0.0]
    obs = SimulatorObservable(identity, size(state), name = :obs)
    SBI.initialize!(data, obs, state)
    @test SBI.getvalue(data, obs) == state
    SBI.observe!(data, obs, state)
    observed_state = SBI.getvalue(data, obs)
    @test observed_state == state
    # with observable mapping function
    data = SimulationData()
    state = [-1.0, 2.0]
    obs = SimulatorObservable(x -> x.^2, size(state), name = :obs)
    SBI.initialize!(data, obs, state)
    SBI.observe!(data, obs, state)
    observed_state = SBI.getvalue(data, obs)
    @test observed_state == state.^2
end

@testset "TimeSampled" begin
    obsfunc(state) = state.x
    t0 = DateTime(2000,1,1)
    savepoints = t0+Day(1):Day(1):DateTime(2001,1,1)
    # case 1: scalar state
    data = SimulationData()
    state = (x = 0.0,)
    buffered_observable = SimulatorObservable(
        obsfunc,
        size(state.x),
        name = :testobs,
        output = TimeSampled(t0, savepoints; samplerate=Hour(1))
    )
    @test buffered_observable.output.tsave == collect(savepoints)
    @test all(diff(buffered_observable.output.tsample) .== Hour(1))
    SBI.initialize!(data, buffered_observable, state)
    # observable holds no state; the transient sample buffer lives in `data`
    @test SBI.has_buffer(data, :testobs)
    @test typeof(SBI.get_buffer(data, :testobs)) <: SBI.DataSeries
    # update observable at each sample point
    for t in t0:Hour(1):savepoints[end]
        state = (x = 1.0,)
        SBI.observe!(data, buffered_observable, state)
    end
    obs_result = SBI.getvalue(data, buffered_observable)
    # we save 1.0 at each step, so average should always be 1
    @test all(obs_result .≈ 1.0)
    @test length(obs_result) == length(savepoints)
    # case 2: vector state
    data = SimulationData()
    state = (x = ones(10),)
    buffered_observable = SimulatorObservable(
        obsfunc,
        size(state.x),
        output = TimeSampled(t0, savepoints; samplerate=Hour(1)),
        name = :testobs
    )
    SBI.initialize!(data, buffered_observable, state)
    @test SBI.has_buffer(data, :testobs)
    @test typeof(SBI.get_buffer(data, :testobs)) <: SBI.DataSeries
end
