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
the number of dimensions. Default implementation returns `1:Nᵢ` in each dimension where `Nᵢ`
is the size of dimension `i`.
"""
coordinates(obs::Observable) = coordinates(size(obs))

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

Base.nameof(obs::SimulatorObservable) = obs.name

Base.size(obs::SimulatorObservable) = map(length, obs.coords)

function Base.show(io::IO, mime::MIME"text/plain", obs::SimulatorObservable{N,outputType}) where {N,outputType<:SimulatorOutput}
    println(io, "$(nameof(outputType)) SimulatorOsbervable $(obs.name) with $N $(N > 1 ? "dimensions" : "dimension")")
    show(io, mime, obs.coords)
end

coordinates(obs::SimulatorObservable) = obs.coords

mutable struct Transient <: SimulatorOutput
    state::Union{Missing,AbstractVecOrMat}
end

"""
    SimulatorObservable(name::Symbol, f::Function, coords::Tuple)

Constructs a `Transient` observable with state mapping function `f` and coordinates `coords`.
"""
function SimulatorObservable(name::Symbol, f::Function, coords::Tuple)
    ds = coordinates(coords)
    return SimulatorObservable(name, f, Transient(missing), ds)
end

initialize!(obs::SimulatorObservable{N,Transient}, state) where {N} = observe!(obs, state)

function observe!(obs::SimulatorObservable{N,Transient}, state) where {N}
    out = reshape(collect(obs.obsfunc(state)), size(obs))
    obs.output.state = out
    return out
end

function getvalue(obs::SimulatorObservable{N,Transient}, ::Type{T}=Any) where {N,T}
    return obs.output.state
end

function setvalue!(obs::SimulatorObservable{N,Transient}, value) where {N}
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

const TimeSampledObservable{N,T} = SimulatorObservable{N,T} where {N,T<:TimeSampled}

"""
    SimulatorObservable(
        name::Symbol,
        obsfunc,
        t0::tType,
        tsave::AbstractVector{tType},
        output_shape_or_coords::Tuple;
        reducer=mean,
        samplerate=Hour(3),
    ) where {tType}

Constructs a `TimeSampled` observable which iteratively samples and stores outputs on each call to `observe!`.
"""
function SimulatorObservable(
    name::Symbol,
    obsfunc,
    t0::tType,
    tsave::AbstractVector{tType},
    coords::Tuple;
    reducer=mean,
    samplerate=Hour(3),
) where {tType}
    return SimulatorObservable(
        name,
        obsfunc,
        TimeSampled(t0, tsave; reducer, samplerate),
        (coordinates(coords)..., Ti(tsave)),
    )
end

"""
    sampletimes(::TimeSampledObservable)

Return the time points at which the simulator should be sampled in order to compare to
observations. Note that this may not exactly correspond to the observation time points;
e.g. mean annual ground temperature observations would require the simulator to be sampled
at appropriate intervals relative to the forcing. The implementation of `SimulatorObservable` is thus
responsible for computing and storing the model state at each sample time.
"""
sampletimes(obs::TimeSampledObservable) = obs.output.tsample

"""
    savetimes(::TimeSampledObservable)

Return the time points at which simulator outputs will be saved.
"""
savetimes(obs::TimeSampledObservable) = obs.output.tsave


"""
    initialize!(obs::TimeSampledObservable, state)

Initialize the given time-sampled observable with the initial simulator state. Note that this method
checks whether the output of `obsfunc` actually matches the declared size `size(obs)` and will error
if they do not match.
"""
function initialize!(obs::TimeSampledObservable, state)
    # Y = _coerce(obs.obsfunc(state), size(obs)[1:end-1])
    obs.output.buffer = []
    obs.output.output = []
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
        push!(obs.output.output, obs.output.reducer(obs.output.buffer))
        # empty buffer
        resize!(obs.output.buffer, 0)
    end
    # get observable vector at current state
    Y_t = _coerce(obs.obsfunc(state), size(obs)[1:end-1])
    push!(obs.output.buffer, Y_t)
    # update cached time
    obs.output.sampleidx += 1
    return Y_t
end

function getvalue(obs::TimeSampledObservable, ::Type{TT}=Float64) where {TT}
    @assert !isnothing(obs.output.buffer) "observable not yet initialized"
    out = reduce(hcat, obs.output.output)
    # drop first dimension if it is of length 1
    return size(out,1) == 1 ? dropdims(out, dims=1) : out
end

function setvalue!(obs::TimeSampledObservable, values::AbstractMatrix)
    obs.output.buffer = []
    obs.output.output = collect(eachcol(values))
end

function setvalue!(obs::TimeSampledObservable, values::AbstractVector{<:AbstractVector})
    obs.output.buffer = []
    obs.output.output = values
end

unflatten(obs::TimeSampledObservable, x::AbstractVector) = reshape(x, length(first(obs.output.output)), length(obs.output.output))

_coerce(output::AbstractVector, shape::Dims) = reshape(output, shape)
_coerce(output::Number, ::Tuple{}) = [output] # lift to single element vector
_coerce(output, shape) = error("output of observable function must be a scalar or a vector! expected: $(shape), got $(typeof(output)) with $(size(output))")
