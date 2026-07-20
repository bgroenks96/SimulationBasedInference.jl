using SimulationBasedInference

using Dates
using Test

# lightweight mock simulator carrying a value `x` and a time `t`, for driving observe! manually
struct MockSim{X,T}
    x::X
    t::T
end
SBI.current_time(m::MockSim) = m.t

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
    obsfunc(sim) = sim.x
    t0 = DateTime(2000,1,1)
    savepoints = t0+Day(1):Day(1):DateTime(2001,1,1)

    # case 1: scalar state
    data = SimulationData()
    buffered_observable = SimulatorObservable(
        obsfunc,
        size(0.0),
        name = :testobs,
        output = TimeSampled(t0, savepoints; samplerate=Hour(1))
    )
    @test buffered_observable.output.tsave == collect(savepoints)
    @test all(diff(buffered_observable.output.tsample) .== Hour(1))
    SBI.initialize!(data, buffered_observable, MockSim(0.0, t0))
    @test SBI.has_output(data, :testobs)
    @test typeof(SBI.get_output_buffer(data, :testobs)) <: SBI.DataBuffer
    # update observable at each sample point (time supplied via current_time(sim))
    for t in t0:Hour(1):savepoints[end]
        SBI.observe!(data, buffered_observable, MockSim(1.0, t))
    end
    obs_result = SBI.getvalue(data, buffered_observable)
    # we save 1.0 at each step, so average should always be 1
    @test all(obs_result .≈ 1.0)
    @test length(obs_result) == length(savepoints)

    # case 2: vector state
    data = SimulationData()
    buffered_observable = SimulatorObservable(
        obsfunc,
        size(ones(10)),
        output = TimeSampled(t0, savepoints; samplerate=Hour(1)),
        name = :testobs
    )
    SBI.initialize!(data, buffered_observable, MockSim(ones(10), t0))
    @test SBI.has_output(data, :testobs)
    @test typeof(SBI.get_output_buffer(data, :testobs)) <: SBI.DataBuffer
end

@testset "TimeAggregated" begin
    obsfunc(sim) = sim.x
    t0 = DateTime(2000,1,1)
    daily_savepoints = collect(t0+Day(1):Day(1):DateTime(2002,1,1))
    yearly_savepoints = [DateTime(2001,1,1), DateTime(2002,1,1)]

    make_daily(; reducer=mean) = SimulatorObservable(
        obsfunc,
        size(0.0),
        name = :daily,
        output = TimeSampled(t0, daily_savepoints; samplerate=Hour(1), reducer),
    )

    # helper: drive only the source over the hourly grid; the aggregate is derived afterwards
    function drive_source!(data, daily; signal = _ -> 1.0)
        SBI.initialize!(data, daily, MockSim(signal(t0), t0))
        for t in t0:Hour(1):last(daily_savepoints)
            SBI.observe!(data, daily, MockSim(signal(t), t))
        end
        return data
    end

    # construction / metadata
    daily = make_daily()
    yearly = TimeAggregatedObservable(daily, yearly_savepoints; name=:yearly)
    @test yearly.output isa TimeAggregated
    @test yearly.output.source == :daily
    @test SBI.savetimes(yearly) == yearly_savepoints
    @test SBI.sampletimes(yearly) == []                # not sampled during the solve
    @test yearly.coords == daily.coords
    # yearly save times must be a subset of the source save times
    @test_throws AssertionError TimeAggregatedObservable(daily, [DateTime(2000,6,15,12)])
    # source must be a TimeSampled observable
    @test_throws MethodError TimeAggregatedObservable(
        SimulatorObservable(identity, size(0.0); name=:tr), yearly_savepoints)

    # case 1: constant field -> daily means and yearly mean are all 1.0 (lazy getvalue)
    data = SimulationData()
    d1 = make_daily(); y1 = TimeAggregatedObservable(d1, yearly_savepoints; name=:yearly)
    drive_source!(data, d1)
    yearly_result = SBI.getvalue(data, y1)
    @test length(yearly_result) == length(yearly_savepoints)
    @test all(yearly_result .≈ 1.0)
    @test all(SBI.getvalue(data, d1) .≈ 1.0)

    # case 2: per-slice transform applied before aggregating (identity -> ×2)
    data = SimulationData()
    d2 = make_daily(); y2 = TimeAggregatedObservable(d2, yearly_savepoints; name=:yearly, transform = x -> 2 .* x)
    drive_source!(data, d2)
    @test all(SBI.getvalue(data, y2) .≈ 2.0)

    # case 3: sum reducer at both levels -> aggregation partitions without loss/duplication
    data = SimulationData()
    d3 = make_daily(reducer=sum); y3 = TimeAggregatedObservable(d3, yearly_savepoints; name=:yearly, reducer=sum)
    drive_source!(data, d3; signal = t -> 2.0)
    SBI.getvalue(data, y3)  # materialize
    daily_total  = sum(v -> v[1], SBI.getoutput(data, :daily))
    yearly_total = sum(v -> v[1], SBI.getoutput(data, :yearly))
    @test yearly_total ≈ daily_total

    # case 4: finalize! eagerly materializes the aggregate as a stored output
    data = SimulationData()
    d4 = make_daily(); y4 = TimeAggregatedObservable(d4, yearly_savepoints; name=:yearly)
    drive_source!(data, d4)
    SBI.initialize!(data, y4, MockSim(1.0, t0))
    @test length(SBI.getoutput(data, :yearly)) == 0   # not populated by the solve loop
    SBI.finalize!(data, (d4, y4))
    @test length(SBI.getoutput(data, :yearly)) == length(yearly_savepoints)
    @test all(v -> v[1] ≈ 1.0, SBI.getoutput(data, :yearly))
end
