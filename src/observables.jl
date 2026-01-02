abstract type SimulatorOutput{T} end

"""
    Observable{outputType<:SimulatorOutput}

Base type for observables with the given `outputType`.
"""
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
    getvalue(::Observable, ::Type{T}=Any) where {T}

Retreive the obsevable at all coordinates.
"""
getvalue(obs::Observable, ::Type{T}=Any) where {T} = error("not implemented for observable of type $(typeof(obs))")

"""
    setvalue!(obs::Observable, value)

Overwrites the value of this observable. The type of `value` will depend on the type of the observable.
This should generally only be used for testing and emulation purposes.
"""
setvalue!(obs::Observable, value) = error("not implemented for observable of type $(typeof(obs))")

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
    SimulatorObservable{N,outputType<:SimulatorOutput,funcType,coordsType} <: Observable{outputType}

Represents a named "observable" that stores output from a simulator. `obsfunc`
defines a mapping from the simulator state to the observed quantity. The type
and implementation of `output` determines how the samples are stored. The simplest
output type is `Transient` which simply maintains a pointer to the last observed
output.
"""
struct SimulatorObservable{N,outputType<:SimulatorOutput,funcType,coordsType<:Tuple{Vararg{Dimension,N}}} <: Observable{outputType}
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

Simple output type that stores a transient reference to an arbitrary state variable. The reference is
overwritten on each subsequent call to `observe!`.
"""
mutable struct Transient{T} <: SimulatorOutput{T}
    value::Union{Nothing, T}
end

"""
    SimulatorObservable(func, coords::Tuple; output::SimulatorOutput = Transient{T}(nothing), name::Symbol = :obs)

Constructs an observable based on the given function `func(state)::T` and `output` type. Defaults to `Transient`
output which simply saves the last observed value of `func`. The coordinates `coords` describe the shape of the output.
"""
function SimulatorObservable(func, coords::Tuple; output::SimulatorOutput = Transient{Any}(nothing), name::Symbol = :obs)
    ds = coordinates(coords)
    return SimulatorObservable(name, func, output, ds)
end

initialize!(obs::SimulatorObservable{N, <:Transient}, state) where {N} = observe!(obs, state)

function observe!(obs::SimulatorObservable{N, <:Transient}, state) where {N}
    out = _coerce(obs.obsfunc(state), size(obs))
    obs.output.value = out
    return out
end

function getvalue(obs::SimulatorObservable{N, <:Transient}) where {N}
    data = obs.output.value
    coords = coordinates(obs)
    return DimArray(data, coords)
end

function setvalue!(obs::SimulatorObservable{N, <:Transient}, value) where {N}
    obs.output.value = value
end

"""
    TimeSampled{timeType, storageType, reducerType, converterType} <: SimulatorOutput

`SimulatorOutput` which buffers samples taken from the simulator at preset times and applies a reduction operation at
(lower frequency) save times. A simple example would be a windowed average or resampling operation that saves averages
over higher frequency samples.
"""
mutable struct TimeSampled{timeType, outputType, storageType<:SimulationData{timeType, outputType}, reducerType, converterType} <: SimulatorOutput{outputType}
    tspan::NTuple{2,timeType}
    tsample::Vector{timeType} # sample times
    tsave::Vector{timeType} # save times
    tconvert::converterType # time converter
    reducer::reducerType # reducer function
    storage::storageType
    buffer::Union{Nothing, Vector{outputType}}
    sampleidx::Int
end

"""
    TimeSampled(
        t0::tType,
        tsave::AbstractVector{tType};
        reducer=mean,
        samplerate=default_sample_rate(tsave),
    ) where {tType}

Constructs a `TimeSampled` simulator output which iteratively samples and stores outputs on each call to `observe!`.
"""
function TimeSampled(
    t0::timeType,
    tsave::AbstractVector{timeType};
    time_converter = convert,
    reducer=mean,
    samplerate=default_sample_rate(tsave),
    output_type=Any,
    storage::SimulationData=SimulationArrayStorage(; input_type=timeType, output_type),
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
    return TimeSampled(extrema(tsample), tsample, collect(tsave), time_converter, reducer, storage, nothing, 1)
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

"""
    savetimes(::TimeSampledObservable)
    savetimes(::Type{T}, obs::TimeSampledObservable) where {T}

Return the time points at which simulator outputs will be saved. If a time type `T` is specified,
the sample times are converted to `T` before returning.
"""
savetimes(obs::TimeSampledObservable) = obs.output.tsave
savetimes(::Type{T}, obs::TimeSampledObservable) where {T} = map(t -> obs.output.tconvert(T, t), savetimes(obs))
savetimes(::SimulatorObservable) = []

default_sample_rate(ts::AbstractVector) = minimum(diff(ts))

"""
    initialize!(obs::TimeSampledObservable, state)

Initialize the given time-sampled observable with the initial simulator state. Note that this method
checks whether the output of `obsfunc` actually matches the declared size `size(obs)` and will error
if they do not match.
"""
function initialize!(obs::TimeSampledObservable, state)
    # Y = _coerce(obs.obsfunc(state), size(obs)[1:end-1])
    storage = obs.output.storage
    clear!(storage)
    obs.output.buffer = similar(storage.outputs, 0)
    obs.output.sampleidx = 1
    return nothing
end

function observe!(obs::TimeSampledObservable, state)
    @assert !isnothing(obs.output.buffer) "observable not yet initialized"
    inbounds = obs.output.sampleidx <= length(obs.output.tsample)
    t = inbounds ? obs.output.tsample[obs.output.sampleidx] : obs.output.tsample[end]
    # find index of time point
    idx = searchsorted(obs.output.tsave, t)
    # get observable vector at current state
    Y_t = _coerce(obs.obsfunc(state), size(obs)[1:end-1])
    push!(obs.output.buffer, Y_t)
    # if t ∈ save points, compute and store reduced output
    if first(idx) == last(idx) && inbounds && length(obs.output.buffer) > 0
        store!(obs.output.storage, t, obs.output.reducer(obs.output.buffer))
        # empty buffer
        resize!(obs.output.buffer, 0)
    end
    # update cached time
    obs.output.sampleidx += 1
    return Y_t
end

function getvalue(obs::TimeSampledObservable)
    @assert !isnothing(obs.output.buffer) "observable not yet initialized"
    @assert length(obs.output.storage) > 0 "output buffer is empty; check for errors in the model evaluation"
    outputs = getoutputs(obs.output.storage)
    # time is always the last coordinate of the observable (excluding batch dimension)
    t_idx = length(size(obs))
    # get first output
    y0 = first(outputs)
    out = foldl(outputs, init=similar(y0, tupleinsert(size(y0), t_idx, 0))) do out, yᵢ
        cat(out, reshape(yᵢ, tupleinsert(size(yᵢ), t_idx, 1)), dims=t_idx)
    end
    coords = coordinates(obs)
    darr = DimArray(reshape(out, size(obs)), coords)
    singleton_dims = filter(c -> length(c) == 1, coords)
    return dropdims(darr, dims=singleton_dims)
end

function setvalue!(obs::TimeSampledObservable, values::AbstractArray)
    @assert size(values) == size(obs) "shape of values $(size(values)) does not match that of the observable $(size(obs))"
    resize!(obs.output.buffer, 0)
    clear!(obs.output.storage)
    ts = savetimes(obs)
    for (i, vals) in enumerate(eachslice(values, dims=length(size(values))))
        store!(obs.output.storage, ts[i], vals)
    end
end

setvalue!(obs::TimeSampledObservable, values::AbstractVector{<:AbstractVector}) = setvalue!(obs, reduce(hcat, values))

unflatten(obs::TimeSampledObservable, x::AbstractVector) = reshape(x, length(first(obs.output.storage)), length(obs.output.storage))

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
