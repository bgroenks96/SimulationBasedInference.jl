"""
Base type for simulator observable output handlers.
"""
abstract type SimulatorOutput{T} end

"""
    Observable{outputType<:SimulatorOutput}

Base type for observables with the given `outputType`.
"""
abstract type Observable{outputType<:SimulatorOutput} end

# Observable methods

"""
    initialize!(data::SimulationData, ::Observable, state)

Initialize the `Observable` from the given simulator state, allocating any required storage
(output and transient sample buffers) from the given [`SimulationData`](@ref).
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
`value` will depend on the type of the observable.
"""
setvalue!(::SimulationData, obs::Observable, value) = error("not implemented for observable of type $(typeof(obs))")

"""
    coordinates(obs::Observable)

Retrieves coordinates for each dimension of the observables as a `Tuple` with length matching
the number of dimensions.
"""
coordinates(obs::Observable) = error("not implemented for osbervable of type $(typeof(obs))")

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

# Base methods

Base.size(obs::Observable) = map(length, coordinates(obs))

Base.nameof(obs::SimulatorObservable) = obs.name

function Base.show(io::IO, mime::MIME"text/plain", obs::SimulatorObservable{N,outputType}) where {N,outputType<:SimulatorOutput}
    println(io, "$(nameof(outputType)) SimulatorOsbervable $(obs.name) with $N $(N > 1 ? "dimensions" : "dimension")")
    show(io, mime, obs.coords)
end

# Output types

"""
    Transient{T} <: SimulatorOutput

Simple output type that retains only the last observed value of the observable function. The
value itself is stored in the [`SimulationData`](@ref).
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
    return with_output_buffer(data, obs.name) do buffer
        # evaluate obsfunc on current state
        out = _coerce(obs.obsfunc(state), size(obs))
        # drop any existing data and store the current value
        empty!(buffer)
        store!(buffer, out)
        return out
    end
end

function getvalue(data::SimulationData, obs::SimulatorObservable{N, <:Transient}) where {N}
    values = getoutput(data, obs.name)
    @assert length(values) > 0 "observable $(obs.name) has not yet been observed"
    coords = coordinates(obs)
    return DimArray(last(values), coords)
end

function setvalue!(data::SimulationData, obs::SimulatorObservable{N, <:Transient}, value) where {N}
    return with_output_buffer(data, obs.name) do buffer
        # drop any existing data and store the current value
        empty!(buffer)
        store!(buffer, value)
        return value
    end
end

"""
    TimeSampled{timeType, storageType, reducerType, converterType} <: SimulatorOutput

`SimulatorOutput` which buffers samples taken from the simulator at preset times and applies a reduction operation at
(lower frequency) save times. A simple example would be a windowed average or resampling operation that saves averages
over higher frequency samples.
"""
mutable struct TimeSampled{timeType, outputType, reducerType, converterType} <: SimulatorOutput{outputType}
    tspan::NTuple{2,timeType}
    tsample::Vector{timeType} # sample times
    tsave::Vector{timeType} # save times
    tconvert::converterType # time converter
    reducer::reducerType # reducer function
    sampleidx::Int
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
        extrema(tsample), tsample, collect(tsave), time_converter, reducer, 1
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
    TimeAggregated{timeType, outputType, reducerType, converterType} <: SimulatorOutput

`SimulatorOutput` representing a coarser-scale temporal aggregation of another (`source`)
observable's already-saved outputs. Unlike [`TimeSampled`](@ref), it does **not** sample the raw
simulator state and is not observed during the forward solve; its value is computed by a single
streaming reduction over the source's stored outputs and materialized after the solve completes
(see [`TimeAggregatedObservable`](@ref)). Peak memory is bounded by one aggregation window, not the
full source series.
"""
struct TimeAggregated{timeType, outputType, reducerType, converterType} <: SimulatorOutput{outputType}
    source::Symbol            # name of the source observable
    tsource::Vector{timeType} # source save times (window boundaries), captured at construction
    tsave::Vector{timeType}   # coarse save times (subset of tsource)
    tconvert::converterType   # time converter
    reducer::reducerType      # reduction applied over transformed slices within each window
end

# internal dispatch alias (the public name `TimeAggregatedObservable` is a constructor function)
const TimeAggregatedObs{N,T} = SimulatorObservable{N,T} where {N,T<:TimeAggregated}

coordinates(obs::TimeAggregatedObs) = (obs.coords..., Ti(savetimes(obs)))
savetimes(obs::TimeAggregatedObs) = obs.output.tsave
savetimes(::Type{T}, obs::TimeAggregatedObs) where {T} = map(t -> obs.output.tconvert(T, t), savetimes(obs))

"""
    TimeAggregatedObservable(
        source::TimeSampledObservable,
        tsave::AbstractVector;
        transform = identity,
        reducer = mean,
        coords = source.coords,
        name = Symbol(nameof(source), :_agg),
        output_type = Any,
    )

Constructs an observable that aggregates the already-saved outputs of `source` at the coarser save
times `tsave` (which must be a subset of `savetimes(source)`). For each aggregation window, the
per-slice `transform` is applied to every source time slice and the results are combined with
`reducer`. The value is computed once by a streaming reduction over the source's stored outputs
(materialized after the forward solve, or lazily on first `getvalue`), so the source observable can
still be retained for diagnostics while this coarser observable is compared to observations via a
likelihood.

`transform` defaults to `identity`; when it changes the per-slice shape (e.g. a spatial reduction),
pass the resulting spatial `coords` explicitly.

!!! note "Reducer composition"
    The reduction is applied to the source's saved (already-reduced) values, not the raw state.
    `sum`, `minimum`/`maximum` compose exactly. `mean` composes exactly **only when each coarse
    window contains an equal number of source values** (e.g. daily→yearly); it is biased for
    unequal groups (e.g. monthly→yearly). Choosing a composable reducer is the caller's
    responsibility.
"""
function TimeAggregatedObservable(
    source::TimeSampledObservable,
    tsave::AbstractVector;
    transform = identity,
    reducer = mean,
    coords = source.coords,
    name::Symbol = Symbol(nameof(source), :_agg),
    output_type = Any,
)
    tsource = savetimes(source)
    @assert length(tsave) > 0
    @assert issubset(tsave, tsource) "coarse save times must be a subset of the source's save times"
    tconvert = source.output.tconvert
    timeType = eltype(tsource)
    output = TimeAggregated{timeType, output_type, typeof(reducer), typeof(tconvert)}(
        nameof(source), collect(tsource), collect(tsave), tconvert, reducer,
    )
    # the per-slice transform is stored as the observable's `obsfunc`
    return SimulatorObservable(transform, coords; output, name)
end

"""
    initialize!(data::SimulationData, obs::TimeSampledObservable, state; handle)

Initialize the given time-sampled observable with the initial simulator state. A storage
handle **must** be provided since scratch storage is now a feature of handles only. The
handle will be stored in the output object for use during observe! calls.
"""
function initialize!(data::SimulationData, obs::TimeSampledObservable, state)
    output_buffer = get_output_buffer(data, nameof(obs))
    empty!(output_buffer)
    scratch_buffer = get_scratch_buffer(data, nameof(obs))
    empty!(scratch_buffer)
    obs.output.sampleidx = 1
    return nothing
end

"""
    observe!(data::SimulationData, obs::TimeSampledObservable, state)

Observe the given time-sampled observable from the current simulator state. This method is called
at each integration step to extract the observable value and store it in the scratch storage. The
stored values are then reduced according to the reducer function specified when creating the
observable. **Requires** that initialize! was called with a valid handle parameter.
"""
function observe!(data::SimulationData, obs::TimeSampledObservable, state)
    output_buffer = get_output_buffer(data, nameof(obs))
    scratch_buffer = get_scratch_buffer(data, nameof(obs))
    inbounds = obs.output.sampleidx <= length(obs.output.tsample)
    t = inbounds ? obs.output.tsample[obs.output.sampleidx] : obs.output.tsample[end]
    # find index of time point
    idx = searchsorted(obs.output.tsave, t)
    # get observable vector at current state
    Y_t = _coerce(obs.obsfunc(state), size(obs)[1:end-1])
    store!(scratch_buffer, Y_t)
    # if t ∈ save points, compute and store reduced output
    if first(idx) == last(idx) && inbounds && length(scratch_buffer) > 0
        store!(output_buffer, obs.output.reducer(scratch_buffer))
        # empty scratch
        empty!(scratch_buffer)
    end
    # update cached time
    obs.output.sampleidx += 1
    return Y_t
end

# Assemble a time series of stored spatial slices into a `DimArray` with a trailing time axis,
# dropping singleton dimensions. Shared by `TimeSampled` and `TimeAggregated` observables.
function _build_timeseries(data::SimulationData, obs::SimulatorObservable)
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

getvalue(data::SimulationData, obs::TimeSampledObservable) = _build_timeseries(data, obs)

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

# --- TimeAggregated observable methods ---

# The aggregated observable is not sampled during the solve; its value is derived from the source's
# stored outputs. `initialize!` clears any stale cache; `observe!` is a no-op.
function initialize!(data::SimulationData, obs::TimeAggregatedObs, state)
    empty!(get_output_buffer(data, nameof(obs)))
    empty!(get_scratch_buffer(data, nameof(obs)))
    return nothing
end

observe!(::SimulationData, ::TimeAggregatedObs, state) = nothing

"""
    _aggregate!(data::SimulationData, obs::TimeAggregatedObs)

Materialize the aggregated observable by a single streaming pass over the source observable's stored
outputs: apply the per-slice transform (`obs.obsfunc`) to each source slice, accumulate into a
window buffer, and reduce into the output buffer at each coarse save time. Peak memory is one window.
"""
function _aggregate!(data::SimulationData, obs::TimeAggregatedObs)
    out = get_output_buffer(data, nameof(obs))
    empty!(out)
    src = get_output_buffer(data, obs.output.source)
    n = length(src)
    @assert n > 0 "source observable :$(obs.output.source) for :$(nameof(obs)) has no stored output"
    scratch = get_scratch_buffer(data, nameof(obs))
    empty!(scratch)
    tsource = obs.output.tsource
    tsave = obs.output.tsave
    slice_shape = size(obs)[1:end-1]
    for i in 1:n
        t = i <= length(tsource) ? tsource[i] : tsource[end]
        # apply the per-slice transform, then buffer the result
        y = _coerce(obs.obsfunc(src[i]), slice_shape)
        store!(scratch, y)
        # if t is a coarse save point, reduce the window and store
        idx = searchsorted(tsave, t)
        if first(idx) == last(idx) && length(scratch) > 0
            store!(out, obs.output.reducer(scratch))
            empty!(scratch)
        end
    end
    empty!(scratch)
    return nothing
end

function getvalue(data::SimulationData, obs::TimeAggregatedObs)
    out = get_output_buffer(data, nameof(obs))
    # lazily materialize if not already computed (e.g. outside the solver `finalize` path)
    length(out) == 0 && _aggregate!(data, obs)
    return _build_timeseries(data, obs)
end

function setvalue!(data::SimulationData, obs::TimeAggregatedObs, values::AbstractArray)
    @assert size(values) == size(obs) "shape of values $(size(values)) does not match that of the observable $(size(obs))"
    out = get_output_buffer(data, obs.name)
    empty!(out)
    for vals in eachslice(values, dims=length(size(values)))
        store!(out, vals)
    end
    return values
end
setvalue!(data::SimulationData, obs::TimeAggregatedObs, values::AbstractVector{<:AbstractVector}) = setvalue!(data, obs, reduce(hcat, values))

unflatten(obs::TimeAggregatedObs, x::AbstractVector) = reshape(x, prod(size(obs)[1:end-1]), length(savetimes(obs)))

"""
    finalize!(data::SimulationData, observables)

Materialize any [`TimeAggregatedObservable`](@ref)s from their sources' stored outputs. Called at the
end of a forward solve so that aggregated observables are persisted like any other output (and thus
available to storage/likelihood retrieval paths). A no-op for non-aggregated observables.
"""
function finalize!(data::SimulationData, observables)
    for obs in observables
        obs isa TimeAggregatedObs && _aggregate!(data, obs)
    end
    return nothing
end

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
