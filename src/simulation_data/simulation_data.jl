include("storage_backend.jl")
include("data_series.jl")

############################################################
# Single-simulation view
############################################################

"""
    SimulationData{B<:StorageBackend}

A view of the data for **one** simulation held in a [`StorageBackend`](@ref): its input (the
parameters `θ`), a mutable metadata dictionary, the named persistent **output** series (one
per observable), and a namespace of **transient** scratch buffers used by observables during
a solve. A `SimulationData` is a lightweight `(backend, index)` handle — all state lives in
the backend. Observables and likelihoods are stateless and operate on a `SimulationData`
passed as their first argument.

Construct a standalone, in-memory `SimulationData` with `SimulationData()`; obtain one backed
by a shared store from `allocate!`/`getindex` on a [`SimulationDataSet`](@ref).
"""
struct SimulationData{B<:StorageBackend}
    backend::B
    index::Int
end

function SimulationData(; input=Float64[], metadata=(;))
    backend = InMemoryStorage()
    i = allocate!(backend, input; metadata...)
    return SimulationData(backend, i)
end

getinputs(data::SimulationData) = getinputs(data.backend, data.index)
setinputs!(data::SimulationData, x) = (setinputs!(data.backend, data.index, x); data)
getmetadata(data::SimulationData) = getmetadata(data.backend, data.index)

output_names(data::SimulationData) = output_names(data.backend, data.index)

"""
    getdata(data::SimulationData, name::Symbol)

Return a [`DataSeries`](@ref) view of the persistent output series for observable `name`,
creating it if necessary.
"""
function getdata(data::SimulationData, name::Symbol)
    ensure_output!(data.backend, data.index, name)
    return DataSeries{:output}(data.backend, data.index, name)
end

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
getoutput(data::SimulationData, name::Symbol) = collect(getdata(data, name))

"""
    getoutputs(data::SimulationData)

Return a `NamedTuple` mapping each observable name to its collected output sequence
(persistent series only; transient scratch buffers are excluded).
"""
getoutputs(data::SimulationData) = (; (nm => getoutput(data, nm) for nm in output_names(data))...)

"""
    create_scratch!(data::SimulationData, key::Symbol=:buffer)

Create (and reset) a NEW transient scratch buffer keyed by `key` and return a
[`DataSeries`](@ref) view of it. Does not touch any persistent output series.
"""
function create_scratch!(data::SimulationData, key::Symbol=:buffer)
    ensure_scratch!(data.backend, data.index, key)
    empty_scratch!(data.backend, data.index, key)
    return DataSeries{:scratch}(data.backend, data.index, key)
end

get_scratch(data::SimulationData, key::Symbol=:buffer) = DataSeries{:scratch}(data.backend, data.index, key)
has_scratch(data::SimulationData, key::Symbol) = has_scratch(data.backend, data.index, key)

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

backend(dataset::SimulationDataSet) = dataset.backend

Base.length(dataset::SimulationDataSet) = num_simulations(dataset.backend)
Base.firstindex(::SimulationDataSet) = 1
Base.lastindex(dataset::SimulationDataSet) = length(dataset)
Base.isempty(dataset::SimulationDataSet) = length(dataset) == 0
Base.getindex(dataset::SimulationDataSet, i::Integer) = SimulationData(dataset.backend, i)

"""
    allocate!(dataset::SimulationDataSet, input=Float64[]; metadata...)

Reserve a fresh simulation in the backing store (with the given `input`) and return a
[`SimulationData`](@ref) view of it. The forward solve writes its outputs directly into this
view.
"""
function allocate!(dataset::SimulationDataSet, input=Float64[]; metadata...)
    i = allocate!(dataset.backend, input; metadata...)
    return SimulationData(dataset.backend, i)
end

"""
    store!(dataset::SimulationDataSet, data::SimulationData; metadata...)

Append an existing `SimulationData` (possibly held in a different backend, e.g. a per-member
forward solve) to `dataset`, copying its input and output series into the backing store and
merging any extra `metadata`.
"""
function store!(dataset::SimulationDataSet, data::SimulationData; metadata...)
    i = allocate!(dataset.backend, getinputs(data); getmetadata(data)..., metadata...)
    target = SimulationData(dataset.backend, i)
    for nm in output_names(data)
        for value in getdata(data, nm)
            store!(target, nm, value)
        end
    end
    return target
end

Base.empty!(dataset::SimulationDataSet) = empty!(dataset.backend)

# iterating a dataset yields (input, outputs, metadata) triples, one per simulation
function Base.iterate(dataset::SimulationDataSet, i::Int=1)
    return i <= length(dataset) ? ((getinputs(dataset, i), getoutputs(dataset, i), getmetadata(dataset, i)), i + 1) : nothing
end

getinputs(dataset::SimulationDataSet, i::Integer) = getinputs(dataset[i])
getoutputs(dataset::SimulationDataSet, i::Integer) = getoutputs(dataset[i])
getmetadata(dataset::SimulationDataSet, i::Integer) = getmetadata(dataset[i])

getinputs(dataset::SimulationDataSet) = [getinputs(dataset, i) for i in 1:length(dataset)]
getoutputs(dataset::SimulationDataSet) = [getoutputs(dataset, i) for i in 1:length(dataset)]
getmetadata(dataset::SimulationDataSet) = [getmetadata(dataset, i) for i in 1:length(dataset)]

"""
    iterations(dataset::SimulationDataSet)

Return the number of distinct `iter` values recorded in the simulations' metadata, i.e. the
number of inference iterations. Returns `0` for an empty dataset and `1` when no `iter`
metadata is present.
"""
function iterations(dataset::SimulationDataSet)
    length(dataset) == 0 && return 0
    return maximum(get(getmetadata(dataset, i), :iter, 1) for i in 1:length(dataset))
end

############################################################
# Out-of-core (JLD2) backend entrypoint
############################################################

const SUPPORTED_DISK_FORMATS = (format"JLD2",)

"""
    OnDiskSimulationDataSet(file::File{format}; kwargs...) where {format}

Construct a disk-backed `SimulationDataSet` whose backend persists each simulation to disk
at the given file `path`. Requires a supported file I/O backend to be loaded, e.g. `JLD2`.
"""
OnDiskSimulationDataSet(file::File{format}, args...; kwargs...) where {format} = error(
    "No disk storage backend loaded for $format. Load the corresponding package for one of the supported formats $SUPPORTED_DISK_FORMATS to enable this backend.",
)
