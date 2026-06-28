abstract type SimulatorOutput{T} end

"""
    Observable{outputType<:SimulatorOutput}

Base type for observables with the given `outputType`.
"""
abstract type Observable{outputType<:SimulatorOutput} end

"""
    initialize!(data::SimulationData, ::Observable, state)

Initialize the `Observable` from the given simulator state, allocating any required storage
(output and transient sample buffers) within the given [`SimulationData`](@ref). Observables
are stateless; all per-simulation storage lives in `data`.
"""
initialize!(::SimulationData, obs::Observable, state) = error("not implemented for observable of type $(typeof(obs))")

"""
    observe!(data::SimulationData, ::Observable, state)

Computes the relevant state variables from the given simulator state and stores them in the
given [`SimulationData`](@ref).
"""
observe!(::SimulationData, obs::Observable, state) = error("not implemented for observable of type $(typeof(obs))")

"""
    getvalue(data::SimulationData, ::Observable)

Retrieve the observable value (at all coordinates) from the given [`SimulationData`](@ref).
"""
getvalue(::SimulationData, obs::Observable) = error("not implemented for observable of type $(typeof(obs))")

"""
    setvalue!(data::SimulationData, obs::Observable, value)

Overwrites the value of this observable in the given [`SimulationData`](@ref). The type of
`value` will depend on the type of the observable. This should generally only be used for
testing and emulation purposes.
"""
setvalue!(::SimulationData, obs::Observable, value) = error("not implemented for observable of type $(typeof(obs))")

"""
    coordinates(obs::Observable)

Retrieves coordinates for each dimension of the observables as a `Tuple` with length matching
the number of dimensions.
"""
coordinates(obs::Observable) = error("not implemented for osbervable of type $(typeof(obs))")

"""
    size(obs::Observable)

Retruns the shape of this observable by evaluating the `length` of each set of coordinates returned by `coordinates(obs)`.
"""
Base.size(obs::Observable) = map(length, coordinates(obs))

"""
    coordinates(dims...)

Converts arguments `dims` to a tuple of coordinate `Dimensions` according to the following rules:

    - Integers `n` are converted to simple step indices `1:n`
    - Vectors are converted to `Dim`s
"""
function coordinates(dims...)
    coord(i::Int, n::Int) = Dim{Symbol(:dim,i)}(1:n)
    coord(i::Int, v::AbstractVector) = Dim{Symbol(:dim,i)}(sort(v))
    coord(::Int, d::Dimension) = d
    return map(coord, Tuple(1:length(dims)), dims)
end
coordinates(dims::Tuple) = coordinates(dims...)
coordinates(::Tuple{}) = coordinates(1)

"""
    SimulatorObservable{N, outputType<:SimulatorOutput, funcType, coordsType} <: Observable{outputType}

Represents a named "observable" that stores output from a simulator. `obsfunc`
defines a mapping from the simulator state to the observed quantity. The type
and implementation of `output` determines how the samples are stored. The simplest
output type is `Transient` which simply maintains a pointer to the last observed
output.
"""
struct SimulatorObservable{
    N, outputType<:SimulatorOutput, funcType, coordsType<:Tuple{Vararg{Dimension,N}}
} <: Observable{outputType}
    name::Symbol
    obsfunc::funcType
    output::outputType
    coords::coordsType
end

coordinates(obs::SimulatorObservable) = obs.coords
coordinates(obs::SimulatorObservable, batch_size::Int) = (obs.coords..., Dim{:ens}(1:batch_size))

Base.nameof(obs::SimulatorObservable) = obs.name

function Base.show(io::IO, mime::MIME"text/plain", obs::SimulatorObservable{N,outputType}) where {N,outputType<:SimulatorOutput}
    println(io, "$(nameof(outputType)) SimulatorOsbervable $(obs.name) with $N $(N > 1 ? "dimensions" : "dimension")")
    show(io, mime, obs.coords)
end

"""
    Transient{T} <: SimulatorOutput

Simple output type that retains only the last observed value of the observable function. The
value itself is stored in the [`SimulationData`](@ref); `Transient` is a stateless marker.
"""
struct Transient{T} <: SimulatorOutput{T} end

Transient(::Type{T}=Any) where {T} = Transient{T}()

"""
    SimulatorObservable(func, coords::Tuple; output::SimulatorOutput = Transient(), name::Symbol = :obs)

Constructs an observable based on the given function `func(state)::T` and `output` type. Defaults to `Transient`
output which simply saves the last observed value of `func`. The coordinates `coords` describe the shape of the output.
"""
function SimulatorObservable(func, coords::Tuple; output::SimulatorOutput = Transient(), name::Symbol = :obs)
    ds = coordinates(coords)
    return SimulatorObservable(name, func, output, ds)
end

initialize!(data::SimulationData, obs::SimulatorObservable{N, <:Transient}, state) where {N} = observe!(data, obs, state)

function observe!(data::SimulationData, obs::SimulatorObservable{N, <:Transient}, state) where {N}
    out = _coerce(obs.obsfunc(state), size(obs))
    # get output storage
    buffer = get_output_buffer(data, obs.name)
    # drop any existing data and store the current value
    empty!(buffer)
    store!(buffer, out)
    return out
end

function getvalue(data::SimulationData, obs::SimulatorObservable{N, <:Transient}) where {N}
    buffer = get_output_buffer(data, obs.name)
    @assert length(buffer) > 0 "observable $(obs.name) has not yet been observed"
    coords = coordinates(obs)
    return DimArray(last(buffer), coords)
end

function setvalue!(data::SimulationData, obs::SimulatorObservable{N, <:Transient}, value) where {N}
    buffer = get_output_buffer(data, obs.name)
    empty!(buffer)
    store!(buffer, value)
    return value
end

"""
    TimeSampled{timeType, storageType, reducerType, converterType} <: SimulatorOutput

`SimulatorOutput` which buffers samples taken from the simulator at preset times and applies a reduction operation at
(lower frequency) save times. A simple example would be a windowed average or resampling operation that saves averages
over higher frequency samples.
"""
struct TimeSampled{timeType, outputType, reducerType, converterType} <: SimulatorOutput{outputType}
    tspan::NTuple{2,timeType}
    tsample::Vector{timeType} # sample times
    tsave::Vector{timeType} # save times
    tconvert::converterType # time converter
    reducer::reducerType # reducer function
end

"""
    TimeSampled(
        t0::tType,
        tsave::AbstractVector{tType};
        reducer=mean,
        samplerate=default_sample_rate(tsave),
        handle=nothing,  # Optional handle parameter for efficient scratch storage
    ) where {tType}

Constructs a `TimeSampled` simulator output which iteratively samples and stores outputs on each call to `observe!`.
"""
function TimeSampled(
    t0::timeType,
    tsave::AbstractVector{timeType};
    time_converter = convert,
    reducer = mean,
    samplerate = default_sample_rate(tsave),
    output_type = Any,
) where {timeType}
    @assert length(tsave) > 0
    @assert first(tsave) >= t0
    @assert length(tsave) == 1 || minimum(diff(tsave)) >= samplerate "sample frequency must be >= save frequency"
    tsample = [t0]
    for t in tsave
        # append sample points up to next t
        append!(tsample, tsample[end]+samplerate:samplerate:t-samplerate)
        if t > tsample[end]
            # add next t
            push!(tsample, t)
        end
    end
    return TimeSampled{timeType, output_type, typeof(reducer), typeof(time_converter)}(
        extrema(tsample), tsample, collect(tsave), time_converter, reducer,
    )
end

const TimeSampledObservable{N,T} = SimulatorObservable{N,T} where {N,T<:TimeSampled}

coordinates(obs::TimeSampledObservable) = (obs.coords..., Ti(savetimes(obs)))

"""
    sampletimes(::TimeSampledObservable)
    sampletimes(::Type{T}, obs::TimeSampledObservable) where {T}

Return the time points at which the simulator should be sampled in order to compare to
observations. Note that this may not exactly correspond to the observation time points;
e.g. mean annual ground temperature observations would require the simulator to be sampled
at appropriate intervals relative to the forcing. The implementation of `SimulatorObservable` is thus
responsible for computing and storing the model state at each sample time. If a time type `T` is specified,
the sample times are converted to `T` before returning.
"""
sampletimes(obs::TimeSampledObservable) = obs.output.tsample
sampletimes(::Type{T}, obs::TimeSampledObservable) where {T} = map(t -> obs.output.tconvert(T, t), sampletimes(obs))
sampletimes(::SimulatorObservable) = []
sampletimes(::Type{T}, ::SimulatorObservable) where{T} = []

"""
    savetimes(::TimeSampledObservable)
    savetimes(::Type{T}, obs::TimeSampledObservable) where {T}

Return the time points at which simulator outputs will be saved. If a time type `T` is specified,
the sample times are converted to `T` before returning.
"""
savetimes(obs::TimeSampledObservable) = obs.output.tsave
savetimes(::Type{T}, obs::TimeSampledObservable) where {T} = map(t -> obs.output.tconvert(T, t), savetimes(obs))
savetimes(::SimulatorObservable) = []
savetimes(::Type{T}, ::SimulatorObservable) where{T} = []


default_sample_rate(ts::AbstractVector) = minimum(diff(ts))

"""
    initialize!(data::SimulationData, obs::TimeSampledObservable, state; handle)

Initialize the given time-sampled observable with the initial simulator state. A storage
handle **must** be provided since scratch storage is now a feature of handles only. The
handle will be stored in the output object for use during observe! calls.
"""
function initialize!(data::SimulationData, obs::TimeSampledObservable, state)
    # scratch storage is provided by the handle that `data` wraps during a forward solve;
    # allocate/reset the transient sample buffer and the persistent output buffer
    ensure_scratch!(data.backend, data.index, obs.name)
    empty!(get_scratch_buffer(data, obs.name))
    empty!(get_output_buffer(data, obs.name))
    return nothing
end

"""
    _sample_window(output::TimeSampled, k::Int)

Return the range of (1-based) sample indices belonging to the `k`-th save bucket, i.e. the
samples whose time falls in `(tsave[k-1], tsave[k]]` (with `tsave[0]` taken as the start of
the time span). Indices are into the non-clearing transient sample buffer, which mirrors
`tsample`.
"""
function _sample_window(output::TimeSampled, k::Int)
    save_idx = searchsortedfirst(output.tsample, output.tsave[k])
    prev_idx = k == 1 ? 0 : searchsortedfirst(output.tsample, output.tsave[k-1])
    return (prev_idx + 1):save_idx
end

"""
    observe!(data::SimulationData, obs::TimeSampledObservable, state)

Observe the given time-sampled observable from the current simulator state. This method is called
at each integration step to extract the observable value and store it in the scratch storage. The
stored values are then reduced according to the reducer function specified when creating the
observable. **Requires** that initialize! was called with a valid handle parameter.
"""
function observe!(data::SimulationData, obs::TimeSampledObservable, state)
    output = obs.output
    buffer = get_scratch_buffer(data, obs.name)   # transient sample buffer (via data's handle)
    out = get_output_buffer(data, obs.name)       # persistent output buffer

    # current sample index = number of samples observed so far + 1
    n = length(buffer) + 1
    inbounds = n <= length(output.tsample)
    t = inbounds ? output.tsample[n] : output.tsample[end]
    
    # observe and buffer the current sample
    Y_t = _coerce(obs.obsfunc(state), size(obs)[1:end-1])
    store!(buffer, Y_t)
    
    # if t is the next (not-yet-stored) save point, reduce its sample window and store
    k = length(out) + 1
    if inbounds && k <= length(output.tsave) && t == output.tsave[k]
        window = _sample_window(output, k)
        store!(out, output.reducer(buffer[window]))
    end
    
    return Y_t
end

function getvalue(data::SimulationData, obs::TimeSampledObservable)
    out = get_output_buffer(data, obs.name)
    @assert length(out) > 0 "output for observable $(obs.name) is empty; check for errors in the model evaluation"
    outputs = collect(out)
    # time is always the last coordinate of the observable (excluding batch dimension)
    t_idx = length(size(obs))
    # get first output
    y0 = first(outputs)
    result = foldl(outputs, init=similar(y0, tupleinsert(size(y0), t_idx, 0))) do acc, yᵢ
        cat(acc, reshape(yᵢ, tupleinsert(size(yᵢ), t_idx, 1)), dims=t_idx)
    end
    coords = coordinates(obs)
    darr = DimArray(reshape(result, size(obs)), coords)
    singleton_dims = filter(c -> length(c) == 1, coords)
    return dropdims(darr, dims=singleton_dims)
end

function setvalue!(data::SimulationData, obs::TimeSampledObservable, values::AbstractArray)
    @assert size(values) == size(obs) "shape of values $(size(values)) does not match that of the observable $(size(obs))"
    out = get_output_buffer(data, obs.name)
    empty!(out)
    for vals in eachslice(values, dims=length(size(values)))
        store!(out, vals)
    end
    return values
end

setvalue!(data::SimulationData, obs::TimeSampledObservable, values::AbstractVector{<:AbstractVector}) = setvalue!(data, obs, reduce(hcat, values))

unflatten(obs::TimeSampledObservable, x::AbstractVector) = reshape(x, prod(size(obs)[1:end-1]), length(savetimes(obs)))

"""
    ODEObservable(
        func,
        prob::SciMLBase.AbstractODEProblem,
        coords = size(func(prob.u0, prob.tspan[1]));
        tsave = [prob.tspan[1], prob.tspan[2]],
        name=:u,
        kwargs...
    )

Convenience constructor for `SimulatorObservable` that automatically constructs a `TimeSampled` output
object from the information in the given `AbstractODEProblem`. The observable function should have the
signature `func(u, t)` where `u` is the ODE state and `t` is the timestep. By default, the coordinates
of the output are inferred by evaluating `func` on `u0` and `tspan[1]`.
"""
function ODEObservable(
    func,
    prob::SciMLBase.AbstractODEProblem,
    coords = size(func(prob.u0, prob.tspan[1]));
    tsave = [prob.tspan[1], prob.tspan[2]],
    name=:u,
    kwargs...
)
    output = TimeSampled(prob.tspan[1], tsave; kwargs...)
    return SimulatorObservable(integrator -> func(integrator.u, integrator.t), coords; name, output)
end

_coerce(output, shape) = error("output of observable function must be a scalar or array! expected $shape but got $output")
_coerce(output::Number, ::Tuple{}) = [output] # lift to single element vector
function _coerce(output::Number, shape::Dims{1})
    if shape[1] == 1
        return [output]
    else
        error("scalar output does not match expected dimensions $shape")
    end
end
function _coerce(output::AbstractArray{T,N}, shape::Dims{M}) where {T,N,M}
    if N > M && size(output)[1:length(shape)] == shape
        reshape(output, tuple(shape..., :))
    elseif N == M && size(output)[1:length(shape)] == shape
        output
    else
        error("expected: $(shape) or $(tuple(shape..., :)), got $(typeof(output)) with $(size(output))")
    end
end
