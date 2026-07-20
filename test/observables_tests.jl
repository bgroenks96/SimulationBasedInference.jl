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
    @test SBI.has_output(data, :testobs)
    @test typeof(SBI.get_output_buffer(data, :testobs)) <: SBI.DataBuffer
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
    @test SBI.has_output(data, :testobs)
    @test typeof(SBI.get_output_buffer(data, :testobs)) <: SBI.DataBuffer
end

@testset "TimeSampled aggregation" begin
    obsfunc(state) = state.x
    t0 = DateTime(2000,1,1)
    daily_savepoints = collect(t0+Day(1):Day(1):DateTime(2002,1,1))
    yearly_savepoints = [DateTime(2001,1,1), DateTime(2002,1,1)]

    make_daily(; reducer=mean) = SimulatorObservable(
        obsfunc,
        size(0.0),
        name = :daily,
        output = TimeSampled(t0, daily_savepoints; samplerate=Hour(1), reducer),
    )

    # construction / metadata
    daily = make_daily()
    yearly = TimeAggregatedObservable(daily, yearly_savepoints; name=:yearly)
    @test yearly.output.source == :daily
    @test SBI.savetimes(yearly) == yearly_savepoints
    @test SBI.sampletimes(yearly) == daily_savepoints
    @test yearly.coords == daily.coords
    # yearly save times must be a subset of the source save times
    @test_throws AssertionError TimeAggregatedObservable(daily, [DateTime(2000,6,15,12)])

    # helper: drive both observables over the hourly grid on a shared SimulationData
    function drive!(daily, yearly; signal = _ -> 1.0)
        data = SimulationData()
        SBI.initialize!(data, daily, (x = signal(t0),))
        SBI.initialize!(data, yearly, (x = signal(t0),))
        for t in t0:Hour(1):last(daily_savepoints)
            state = (x = signal(t),)
            SBI.observe!(data, daily, state)   # source observed first
            SBI.observe!(data, yearly, state)  # aggregator reads the source's latest save
        end
        return data
    end

    # case 1: constant field -> daily means and yearly mean are all 1.0
    d1 = make_daily()
    y1 = TimeAggregatedObservable(d1, yearly_savepoints; name=:yearly)
    data = drive!(d1, y1)
    yearly_result = SBI.getvalue(data, y1)
    @test length(yearly_result) == length(yearly_savepoints)
    @test all(yearly_result .≈ 1.0)
    @test all(SBI.getvalue(data, d1) .≈ 1.0)

    # case 2: sum reducer at both levels -> aggregation partitions without loss/duplication
    d_sum = make_daily(reducer=sum)
    y_sum = TimeAggregatedObservable(d_sum, yearly_savepoints; name=:yearly, reducer=sum)
    data = drive!(d_sum, y_sum; signal = t -> 2.0)
    daily_total = sum(v -> v[1], SBI.getoutput(data, :daily))
    yearly_total = sum(v -> v[1], SBI.getoutput(data, :yearly))
    @test yearly_total ≈ daily_total

    # ordering: source is placed before its dependent regardless of declared order
    d = make_daily()
    y = TimeAggregatedObservable(d, yearly_savepoints; name=:yearly)
    ordered = SBI.sort_observables((; yearly=y, daily=d))
    @test map(nameof, ordered) == [:daily, :yearly]

    # error cases
    scalar_coords = size(0.0)
    unknown = SimulatorObservable(obsfunc, scalar_coords; name=:a,
        output = TimeSampled(t0, daily_savepoints; samplerate=Hour(1), source=:missing))
    @test_throws ErrorException SBI.sort_observables((; a=unknown))

    transient = SimulatorObservable(identity, scalar_coords; name=:t)
    agg_of_transient = SimulatorObservable(obsfunc, scalar_coords; name=:b,
        output = TimeSampled(t0, daily_savepoints; samplerate=Hour(1), source=:t))
    @test_throws ErrorException SBI.sort_observables((; t=transient, b=agg_of_transient))

    cyc_a = SimulatorObservable(obsfunc, scalar_coords; name=:a,
        output = TimeSampled(t0, daily_savepoints; samplerate=Hour(1), source=:b))
    cyc_b = SimulatorObservable(obsfunc, scalar_coords; name=:b,
        output = TimeSampled(t0, daily_savepoints; samplerate=Hour(1), source=:a))
    @test_throws ErrorException SBI.sort_observables((; a=cyc_a, b=cyc_b))
end
