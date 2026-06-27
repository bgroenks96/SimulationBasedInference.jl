include("storage_backend.jl")
include("data_buffer.jl")

"""
    SimulationData{B<:StorageBackend}

A view of the data for a single simulation stored in a [`StorageBackend`](@ref): its input (the
parameters `θ`), a mutable metadata dictionary, the named persistent **output** series (one
per observable), and a namespace of **transient** scratch buffers used by observables during
a solve.
"""
struct SimulationData{B<:StorageBackend}
    backend::B
    index::Int
end

"""
Allocate and construct a new `SimulationData` from the given `backend`, or create a new [`InMemoryStorage`](@ref)
backend if not specified.
"""
function SimulationData(; backend::StorageBackend=InMemoryStorage(), input=nothing, metadata...)
    i = allocate!(backend, input; metadata...)
    return SimulationData(backend, i)
end

getinputs(data::SimulationData) = getinputs(data.backend, data.index)
setinputs!(data::SimulationData, x) = (setinputs!(data.backend, data.index, x); data)
getmetadata(data::SimulationData) = getmetadata(data.backend, data.index)

output_names(data::SimulationData) = output_names(data.backend, data.index)

"""
    get_output_buffer(data::SimulationData, name::Symbol)

Return a [`DataBuffer`](@ref) view of the output data for variable `name`, creating it if necessary.
"""
function get_output_buffer(data::SimulationData, name::Symbol)
    ensure_output!(data.backend, data.index, name)
    return DataBuffer{:output}(data.backend, data.index, name)
end

has_output(data::SimulationData, name::Symbol) = has_output(data.backend, data.index, name)

"""
    store!(data::SimulationData, name::Symbol, value)

Append `value` to the persistent output series for observable `name`.
"""
store!(data::SimulationData, name::Symbol, value) =
    (store_output!(data.backend, data.index, name, value); data)

"""
    getoutput(data::SimulationData, name::Symbol)

Return the collected output sequence (a `Vector`) for observable `name`.
"""
getoutput(data::SimulationData, name::Symbol) = collect(get_output_buffer(data, name))

"""
    getoutputs(data::SimulationData)

Return a `NamedTuple` mapping each observable name to its collected output sequence
(persistent series only; transient scratch buffers are excluded).
"""
getoutputs(data::SimulationData) = (; (nm => getoutput(data, nm) for nm in output_names(data))...)

"""
    create_scratch!(data::SimulationData, name::Symbol=:buffer)

Create (and reset) a NEW transient scratch buffer keyed by `name` and return a
[`DataBuffer`](@ref) view of it. Does not touch any persistent output series.
"""
function create_scratch!(data::SimulationData, name::Symbol=:buffer)
    ensure_scratch!(data.backend, data.index, name)
    empty_scratch!(data.backend, data.index, name)
    return DataBuffer{:scratch}(data.backend, data.index, name)
end

function get_scratch_buffer(data::SimulationData, name::Symbol=:buffer)
    ensure_scratch!(data.backend, data.index, name)
    return DataBuffer{:scratch}(data.backend, data.index, name)
end

has_scratch(data::SimulationData, name::Symbol) = has_scratch(data.backend, data.index, name)

Base.empty!(data::SimulationData) = empty!(data.backend, data.index)

############################################################
# Collection-of-simulations view
############################################################

"""
    SimulationDataSet{B<:StorageBackend}

A view of the whole collection of simulations held in a single [`StorageBackend`](@ref) — the
simulations run during an inference procedure. `SimulationDataSet()` uses an in-memory backend;
`OnDiskSimulationDataSet(path)` (provided by the `SimulationBasedInferenceJLD2Ext` extension)
uses a disk-backed one.

Indexing returns a [`SimulationData`](@ref) view of the `i`-th simulation; iterating yields
`(input, outputs, metadata)` triples.
"""
@kwdef struct SimulationDataSet{B<:StorageBackend}
    backend::B = InMemoryStorage()
end

backend(storage::SimulationDataSet) = storage.backend

Base.length(storage::SimulationDataSet) = num_simulations(storage.backend)
Base.firstindex(::SimulationDataSet) = 1
Base.lastindex(storage::SimulationDataSet) = length(storage)
Base.isempty(storage::SimulationDataSet) = length(storage) == 0
Base.getindex(storage::SimulationDataSet, i::Integer) = SimulationData(storage.backend, i)

"""
    allocate!(storage::SimulationDataSet, input=Float64[]; metadata...)

Reserve a fresh simulation in the backing store (with the given `input`) and return a
[`SimulationData`](@ref) view of it. The forward solve writes its outputs directly into this
view.
"""
function allocate!(storage::SimulationDataSet, input=Float64[]; metadata...)
    i = allocate!(storage.backend, input; metadata...)
    return SimulationData(storage.backend, i)
end

"""
    store!(storage::SimulationDataSet, data::SimulationData; metadata...)

Append an existing `SimulationData` (possibly held in a different backend, e.g. a per-member
forward solve) to `storage`, copying its input and output series into the backing store and
merging any extra `metadata`.
"""
function store!(storage::SimulationDataSet, data::SimulationData; metadata...)
    i = allocate!(storage.backend, getinputs(data); getmetadata(data)..., metadata...)
    target = SimulationData(storage.backend, i)
    for nm in output_names(data)
        for value in get_output_buffer(data, nm)
            store!(target, nm, value)
        end
    end
    return target
end

Base.empty!(storage::SimulationDataSet) = empty!(storage.backend)

# iterating a storage yields (input, outputs, metadata) triples, one per simulation
function Base.iterate(storage::SimulationDataSet, i::Int=1)
    return i <= length(storage) ? ((getinputs(storage, i), getoutputs(storage, i), getmetadata(storage, i)), i + 1) : nothing
end

getinputs(storage::SimulationDataSet, i::Integer) = getinputs(storage[i])
getoutputs(storage::SimulationDataSet, i::Integer) = getoutputs(storage[i])
getmetadata(storage::SimulationDataSet, i::Integer) = getmetadata(storage[i])

getinputs(storage::SimulationDataSet) = [getinputs(storage, i) for i in 1:length(storage)]
getoutputs(storage::SimulationDataSet) = [getoutputs(storage, i) for i in 1:length(storage)]
getmetadata(storage::SimulationDataSet) = [getmetadata(storage, i) for i in 1:length(storage)]

"""
    iterations(storage::SimulationDataSet)

Return the number of distinct `iter` values recorded in the simulations' metadata, i.e. the
number of inference iterations. Returns `0` for an empty storage and `1` when no `iter`
metadata is present.
"""
function iterations(storage::SimulationDataSet)
    length(storage) == 0 && return 0
    return maximum(get(getmetadata(storage, i), :iter, 1) for i in 1:length(storage))
end

############################################################
# Out-of-core (JLD2) backend entrypoint
############################################################

const SUPPORTED_DISK_FORMATS = [format"JLD2"]

"""
    OnDiskSimulationDataSet(file::File{format}; kwargs...) where {format}

Construct a disk-backed `SimulationDataSet` whose backend persists each simulation to disk
at the given file `path`. Requires a supported file I/O backend to be loaded, e.g. `JLD2`.
"""
OnDiskSimulationDataSet(file::File{format}, args...; kwargs...) where {format} = error(
    "No disk storage backend loaded for $format. Load the corresponding package for one of the supported formats $SUPPORTED_DISK_FORMATS to enable this backend.",
)
