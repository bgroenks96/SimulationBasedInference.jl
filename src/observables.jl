abstract type SimulatorOutput end

abstract type Observable{outputType<:SimulatorOutput} end

"""
    initialize!(::Observable, state)

Initialize the `Observable` from the given simulator state.
"""
initialize!(obs::Observable, state) = error("not implemented for observable of type $(typeof(obs))")

"""
    observe!(::Observable, state)

Computes and stores the relevant state variables from the given simulator state.
"""
observe!(obs::Observable, state) = error("not implemented for observable of type $(typeof(obs))")

"""
    retrieve(::Observable, ::Type{T}=Any) where {T}

Retreive the obsevable at all saved time points, assuming all sample times have been stored appropriately.
"""
retrieve(obs::Observable, ::Type{T}=Any) where {T} = error("not implemented for observable of type $(typeof(obs))")

"""
    setvalue!(obs::Observable, value)

Overwrites the value of this observable. The type of `value` will depend on the type of the observable.
This should generally only be used for testing and emulation purposes.
"""
setvalue!(obs::Observable, value) = error("not implemented for observable of type $(typeof(obs))")

"""
    SimulatorObservable{outputType<:SimulatorOutput,funcType} <: Observable{outputType}

Represents a named "observable" that stores output from a simulator. `obsfunc`
defines a mapping from the simulator state to the observed quantity. The type
and implementation of `output` determines how the samples are stored. The simplest
output type is `Transient` which simply maintains a pointer to the last observed
output.
"""
struct SimulatorObservable{outputType<:SimulatorOutput,funcType} <: Observable{outputType}
    name::Symbol
    obsfunc::funcType
    ndims::Int
    output::outputType
end

"""
    SimulatorObservable(name::Symbol, f::Function=identity; ndims::Int=1)

Constructs a `Transient` observable with state mapping function `f`.
"""
SimulatorObservable(name::Symbol, f::Function=identity; ndims::Int=1) = SimulatorObservable(name, f, ndims, Transient(nothing))

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
) where {tType} = SimulatorObservable(name, obsfunc, ndims, TimeSampled(t0, tsave; reducer, samplerate))

Base.nameof(obs::SimulatorObservable) = obs.name

mutable struct Transient <: SimulatorOutput
    state::Union{Nothing,AbstractVector}
end

Base.size(obs::SimulatorObservable{Transient}) = size(obs.output.state)

initialize!(obs::SimulatorObservable{Transient}, state) = observe!(obs, state)

function observe!(obs::SimulatorObservable{Transient}, state)
    out = obs.obsfunc(state)
    @assert ismissing(out) || length(out) == obs.ndims "Expected output of length $(obs.ndims) but got $(length(out))"
    obs.output.state = out
    return out
end

function retrieve(obs::SimulatorObservable{Transient}, ::Type{T}=Any) where {T}
    return obs.output.state
end

function setvalue!(obs::SimulatorObservable{Transient}, value)
    obs.output.state = value
end

"""
    TimeSampled{timeType,reducerType} <: SimulatorOutput

`SimulatorOutput` which buffers samples taken from the simulator at preset times and applies a reduction operation at
(lower frequency) save times. A simple example would be a windowed average or resampling operation that saves averages
over higher frequency samples.
"""
mutable struct TimeSampled{timeType,reducerType} <: SimulatorOutput
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
    reducer=mean,
    samplerate=Hour(3)
) where {tType}
    @assert length(tsave) > 0
    @assert first(tsave) >= t0
    @assert length(tsave) == 1 || minimum(diff(tsave)) >= samplerate "sample frequency must be higher than all saving intervals"
    tsample = [t0]
    for t in tsave
        # append sample points up to next t
        append!(tsample, tsample[end]+samplerate:samplerate:t-samplerate)
        if t > tsample[end]
            # add next t
            push!(tsample, t)
        end
    end
    return TimeSampled(extrema(tsample), tsample, collect(tsave), reducer, nothing, nothing, 1)
end

const TimeSampledObservable{T} = SimulatorObservable{T} where {T<:TimeSampled}

Base.size(obs::TimeSampledObservable) = (obs.ndims, length(obs.output.tsave))

"""
    sampletimes(::DynamicSimulatorObservable)

Return the time points at which the simulator should be sampled in order to compare to
observations. Note that this may not exactly correspond to the observation time points;
e.g. mean annual ground temperature observations would require the simulator to be sampled
at appropriate intervals relative to the forcing. The implementation of `SimulatorObservable` is thus
responsible for computing and storing the model state at each sample time.
"""
sampletimes(obs::TimeSampledObservable) = obs.output.tsample

"""
    savetimes(::DynamicSimulatorObservable)

Return the time points at which simulator outputs will be saved.
"""
savetimes(obs::TimeSampledObservable) = obs.output.tsave

function initialize!(obs::TimeSampledObservable, state)
    Y = obs.obsfunc(state)
    @assert isa(Y, AbstractVector) || isa(Y, Number) "output of observable function must be a scalar or a vector!"
    @assert length(Y) == obs.ndims "Size of observable vector $(length(Y)) does not match declared size $(obs.ndims)"
    obs.output.buffer = typeof(Y)[]
    obs.output.output = Vector{typeof(Y)}(undef, length(obs.output.tsave))
    obs.output.sampleidx = 1
    return nothing
end

function observe!(obs::TimeSampledObservable, state)
    @assert !isnothing(obs.output.buffer) "observable not yet initialized"
    t = obs.output.tsample[obs.output.sampleidx]
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

function retrieve(obs::TimeSampledObservable, ::Type{TT}=Float64) where {TT}
    @assert !isnothing(obs.output.buffer) "observable not yet initialized"
    out = reduce(hcat, obs.output.output)
    # drop first dimension if it is of length 1
    return size(out,1) == 1 ? dropdims(out, dims=1) : out
end

function setvalue!(obs::TimeSampledObservable, values::AbstractMatrix)
    obs.output.output = collect(eachcol(values))
end

function setvalue!(obs::TimeSampledObservable, values::AbstractVector{<:AbstractVector})
    obs.output.output = values
end

unflatten(obs::TimeSampledObservable, x::AbstractVector) = reshape(x, length(first(obs.output.output)), length(obs.output.output))
