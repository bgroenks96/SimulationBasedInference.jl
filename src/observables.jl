abstract type SimulatorObservable end

# SimulatorObservable interface

Base.nameof(obs::SimulatorObservable) = obs.name

"""
    samplepoints(::SimulatorObservable)

Return the time points at which the simulator should be sampled in order to compare to
observations. Note that this may not exactly correspond to the observation time points;
e.g. mean annual ground temperature observations would require the simulator to be sampled
at appropriate intervals relative to the forcing. The implementation of `SimulatorObservable` is thus
responsible for computing and storing the model state at each sample time.
"""
samplepoints(::SimulatorObservable) = error("not implemented")

"""
    init!(::SimulatorObservable, state)

Initialize the `SimulatorObservable` from the given simulator state.
"""
init!(obs::SimulatorObservable, state)

"""
    observe!(::SimulatorObservable, state)

Computes and stores the relevant state variables from the given simulator state.
"""
observe!(::SimulatorObservable, state) = error("not implemented")

"""
    retrieve(::SimulatorObservable, ::Type{T}=Float64) where {T}

Retreive the obsevable at all saved time points, assuming all sample times have been stored appropriately.
"""
retrieve(::SimulatorObservable, ::Type{T}=Float64) where {T} = error("not implemented")

# simple cache structure for BufferedObservable
mutable struct BufferedObservableCache
    output::Union{Nothing,AbstractVector}
    buffer::Union{Nothing,AbstractVector}
    sampleidx::Int
end

"""
    BufferedObservable{timeType,sampleFuncType,reducerType} <: Observable

Observable which buffers samples taken at preset times and applies a reduction operation at (lower frequency) save times.
A simple example would be a windowed average or resampling operation that saves averages over higher frequency samples.
"""
struct BufferedObservable{timeType,sampleFuncType,reducerType} <: SimulatorObservable
    name::Symbol # identifier
    ndims::Int
    tspan::NTuple{2,timeType}
    tsample::Vector{timeType} # sample times
    tsave::Vector{timeType} # save times
    samplefunc::sampleFuncType # sample function: state -> Vector
    reducer::reducerType # reducer function
    cache::BufferedObservableCache
end
function BufferedObservable(name::Symbol, samplefunc, t0::tType, tsave::AbstractVector{tType}; ndims=1, reducer=mean, samplerate=Hour(3)) where {tType}
    @assert length(tsave) > 0
    @assert first(tsave) >= t0
    @assert minimum(diff(tsave)) >= samplerate "sample frequency must be higher than all saving intervals"
    tsample = [t0]
    for t in tsave
        # append sample points up to next t
        append!(tsample, tsample[end]+samplerate:samplerate:t-samplerate)
        # add next t
        push!(tsample, t)
    end
    cache = BufferedObservableCache(nothing, nothing, 1)
    return BufferedObservable(name, ndims, extrema(tsample), tsample, collect(tsave), samplefunc, reducer, cache)
end

Base.size(obs::BufferedObservable) = (obs.ndims, length(obs.tsave))

samplepoints(obs::BufferedObservable) = obs.tsample

function init!(obs::BufferedObservable, state)
    Y = obs.samplefunc(state)
    @assert isa(Y, AbstractVector) || isa(Y, Number) "output of observable function must be a scalar or a vector!"
    @assert length(Y) == obs.ndims "Size of observable vector $(length(Y)) does not match declared size $(obs.ndims)"
    obs.cache.buffer = typeof(Y)[]
    obs.cache.output = Vector{typeof(Y)}(undef, length(obs.tsave))
    obs.cache.sampleidx = 1
    return nothing
end

function observe!(obs::BufferedObservable, state)
    @assert !isnothing(obs.cache) "observable not yet initialized"
    t = obs.tsample[obs.cache.sampleidx]
    if t < first(obs.tsample) || t > last(obs.tsample)
        # do nothing if outside of sampling period
        return nothing
    end
    # find index of time point
    idx = searchsorted(obs.tsave, t)
    # if t ∈ save points, compute and store reduced output
    if first(idx) == last(idx) && length(obs.cache.buffer) > 0
        obs.cache.output[first(idx)] = obs.reducer(obs.cache.buffer)
        # empty buffer
        resize!(obs.cache.buffer, 0)
    end
    # get observable vector at current state
    Y_t = obs.samplefunc(state)
    push!(obs.cache.buffer, Y_t)
    # update cached time
    obs.cache.sampleidx += 1
    return Y_t
end

function retrieve(obs::BufferedObservable, ::Type{TT}=Float64) where {TT}
    @assert !isnothing(obs.cache) "observable not yet initialized"
    if length(obs.cache.output) > 1
        return reduce(hcat, obs.cache.output)
    else
        return first(obs.cache.output)
    end
end

unflatten(obs::BufferedObservable, x::AbstractVector) = reshape(x, length(first(obs.cache.output)), length(obs.cache.output))

"""
    SnapshotObservable(name::Symbol, samplefunc, tspan::NTuple{2,timeType}, samplerate::timeType; ndims=1)

Convenience alias for `BufferedObservable` that saves the output of `samplefunc` at all sample points.
"""
function SnapshotObservable(name::Symbol, samplefunc, tspan::NTuple{2,timeType}, samplerate::timeType; ndims=1) where {timeType}
    tsave = tspan[1]:samplerate:tspan[2]
    return BufferedObservable(name, samplefunc, tspan[1], tsave; reducer=identity, samplerate=samplerate, ndims)
end
