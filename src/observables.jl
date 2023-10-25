abstract type SimulatorOutput end

"""
    SimulatorObservable{outputType<:SimulatorOutput,funcType}

Represents a named "observable" that stores output from a simulator. `obsfunc`
defines a mapping from the simulator state to the observed quantity. The type
and implementation of `output` determines how the samples are stored. The simplest
output type is `Transient` which simply maintains a pointer to the last observed
output.
"""
struct SimulatorObservable{outputType<:SimulatorOutput,funcType}
    name::Symbol
    obsfunc::funcType
    output::outputType
end

"""
    SimulatorObservable(name::Symbol, f::Function=identity)

Constructs a `Transient` observable with state mapping function `f`.
"""
SimulatorObservable(name::Symbol, f::Function=identity) = SimulatorObservable(name, f, Transient(nothing))

"""
    SimulatorObservable(
        name::Symbol,
        obsfunc,
        t0::tType,
        tsave::AbstractVector{tType};
        ndims=1,
        reducer=mean,
        samplerate=Hour(3),
    ) where {tType}

Constructs a `TimeSampled` observable which iteratively samples and stores outputs on each call to `observe!`.
"""
SimulatorObservable(
    name::Symbol,
    obsfunc,
    t0::tType,
    tsave::AbstractVector{tType};
    ndims=1,
    reducer=mean,
    samplerate=Hour(3),
) where {tType} = SimulatorObservable(name, obsfunc, TimeSampled(t0, tsave; ndims, reducer, samplerate))

Base.nameof(obs::SimulatorObservable) = obs.name

mutable struct Transient <: SimulatorOutput
    state
end

"""
    initialize!(::SimulatorObservable, state)

Initialize the `SimulatorObservable` from the given simulator state.
"""
initialize!(obs::SimulatorObservable{Transient}, state) = obs.output.state = obs.obsfunc(state)

"""
    observe!(::SimulatorObservable, state)

Computes and stores the relevant state variables from the given simulator state.
"""
observe!(obs::SimulatorObservable{Transient}, state) = obs.output.state = obs.obsfunc(state)

"""
    retrieve(::SimulatorObservable, ::Type{T}=Any) where {T}

Retreive the obsevable at all saved time points, assuming all sample times have been stored appropriately.
"""
retrieve(obs::SimulatorObservable{Transient}, ::Type{T}=Any) where {T} = obs.output.state

"""
    TimeSampled{timeType,reducerType} <: SimulatorOutput

`SimulatorOutput` which buffers samples taken from the simulator at preset times and applies a reduction operation at
(lower frequency) save times. A simple example would be a windowed average or resampling operation that saves averages
over higher frequency samples.
"""
mutable struct TimeSampled{timeType,reducerType} <: SimulatorOutput
    ndims::Int
    tspan::NTuple{2,timeType}
    tsample::Vector{timeType} # sample times
    tsave::Vector{timeType} # save times
    reducer::reducerType # reducer function
    output::Union{Nothing,AbstractVector}
    buffer::Union{Nothing,AbstractVector}
    sampleidx::Int
end

function TimeSampled(
    t0::tType,
    tsave::AbstractVector{tType};
    ndims=1,
    reducer=mean,
    samplerate=Hour(3)
) where {tType}
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
    return TimeSampled(ndims, extrema(tsample), tsample, collect(tsave), reducer, nothing, nothing, 1)
end

const DynamicSimulatorObservable{T} = SimulatorObservable{T} where {T<:TimeSampled}

Base.size(obs::DynamicSimulatorObservable) = (obs.output.ndims, length(obs.output.tsave))

"""
    samplepoints(::DynamicSimulatorObservable)

Return the time points at which the simulator should be sampled in order to compare to
observations. Note that this may not exactly correspond to the observation time points;
e.g. mean annual ground temperature observations would require the simulator to be sampled
at appropriate intervals relative to the forcing. The implementation of `SimulatorObservable` is thus
responsible for computing and storing the model state at each sample time.
"""
samplepoints(obs::DynamicSimulatorObservable) = obs.output.tsample

function initialize!(obs::DynamicSimulatorObservable, state)
    Y = obs.obsfunc(state)
    @assert isa(Y, AbstractVector) || isa(Y, Number) "output of observable function must be a scalar or a vector!"
    @assert length(Y) == obs.output.ndims "Size of observable vector $(length(Y)) does not match declared size $(obs.output.ndims)"
    obs.output.buffer = typeof(Y)[]
    obs.output.output = Vector{typeof(Y)}(undef, length(obs.output.tsave))
    obs.output.sampleidx = 1
    return nothing
end

function observe!(obs::DynamicSimulatorObservable, state)
    @assert !isnothing(obs.output.buffer) "observable not yet initialized"
    t = obs.output.tsample[obs.output.sampleidx]
    if t < first(obs.output.tsample) || t > last(obs.output.tsample)
        # do nothing if outside of sampling period
        return nothing
    end
    # find index of time point
    idx = searchsorted(obs.output.tsave, t)
    # if t ∈ save points, compute and store reduced output
    if first(idx) == last(idx) && length(obs.output.buffer) > 0
        obs.output.output[first(idx)] = obs.output.reducer(obs.output.buffer)
        # empty buffer
        resize!(obs.output.buffer, 0)
    end
    # get observable vector at current state
    Y_t = obs.obsfunc(state)
    push!(obs.output.buffer, Y_t)
    # update cached time
    obs.output.sampleidx += 1
    return Y_t
end

function retrieve(obs::DynamicSimulatorObservable, ::Type{TT}=Float64) where {TT}
    @assert !isnothing(obs.output.buffer) "observable not yet initialized"
    if length(obs.output.output) > 1
        return reduce(hcat, obs.output.output)
    else
        return first(obs.output.output)
    end
end

unflatten(obs::DynamicSimulatorObservable, x::AbstractVector) = reshape(x, length(first(obs.output.output)), length(obs.output.output))
