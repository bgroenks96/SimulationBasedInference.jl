"""
    StorageBackend

Abstract type for the storage backend that owns all persistent simulation data. There is
exactly one backend per [`SimulationDataSet`](@ref); the higher-level types
([`SimulationData`](@ref), [`DataBuffer`](@ref)) are lightweight views that route all reads
and writes through it. The backend decides where the bytes actually live — in memory
([`InMemoryStorage`](@ref)) or out-of-core (e.g. the JLD2 backend provided by the
`SimulationBasedInferenceJLD2Ext` extension).

Each simulation has an input, a metadata dictionary, and a set of named **output** series.
**Scratch** (transient working) series are not part of the backend; they live on a
[`StorageHandle`](@ref) and are discarded when the handle is closed.

Operations come in two forms:
- a fast path taking a [`StorageHandle`](@ref) as the first argument (file kept open), and
- a slow path taking the backend directly, which auto-opens/closes a handle per call via
  `open(backend) do h ... end` (backward compatible).
"""
abstract type StorageBackend end

"""
    StorageHandle

Abstract base type for the object returned by `open(backend::StorageBackend, index::Integer)`. A handle holds
the active connection to a backend (e.g. an open file or in-memory view) for the duration of a batch of
operations on a specific simulation and owns the **scratch** namespace for that simulation.
Concrete handle types must provide a `scratch::Dict{Symbol,Any}` field; scratch is cleared when the handle is closed.

Use with the do-block form for guaranteed cleanup:

    open(backend, index) do h
        # use handle h for simulation index
    end  # handle closed (and scratch cleared) automatically
"""
abstract type StorageHandle end

"""
Apply `func!(handle)` where `handle = open(backend, i; kwargs...)`, automatically
closing the storage handle after `func!` returns.
"""
function with_storage(func!, backend::StorageBackend, i; kwargs...)
    handle = open(backend, i; kwargs...)
    try
        return func!(handle)
    finally
        close(handle)
    end
end

"""
    InMemoryStorage{inputType,outputType,metadataType} <: StorageBackend

In-memory storage backend. Simulation inputs, output series elements, and metadata values are
strongly typed as `inputType`, `outputType`, and `metadataType` respectively. The default
constructor `InMemoryStorage()` uses `Any` for all three.
"""
mutable struct InMemoryStorage{inputType,outputType,metadataType} <: StorageBackend
    inputs::Vector{inputType}
    metadata::Vector{OrderedDict{Symbol,metadataType}}
    outputs::Vector{OrderedDict{Symbol,Vector{outputType}}}
end

InMemoryStorage{I,O,M}(N::Int = 0) where {I,O,M} = InMemoryStorage{I,O,M}(
    Vector{I}(undef, N),
    Vector{OrderedDict{Symbol,M}}(undef, N),
    Vector{OrderedDict{Symbol,Vector{O}}}(undef, N),
)

InMemoryStorage() = InMemoryStorage{Any,Any,Any}()

descriptor(backend::InMemoryStorage) = (type=typeof(backend), num_simulations=num_simulations(backend),)
from_descriptor(::Type{<:InMemoryStorage}, desc::NamedTuple) = desc.type(desc.num_simulations)

# --- backend operations (slow path - auto-open/close per simulation) ---

num_simulations(backend::InMemoryStorage) = length(backend.inputs)
allocate!(backend::InMemoryStorage, input; metadata...) = open(backend, length(backend.inputs) + 1) do h; allocate!(h, input; metadata...); end
getinputs(backend::InMemoryStorage, i::Integer) = open(backend, i) do h; getinputs(h); end
setinputs!(backend::InMemoryStorage, i::Integer, x) = (open(backend, i) do h; setinputs!(h, x); end; backend)
getmetadata(backend::InMemoryStorage, i::Integer) = open(backend, i) do h; getmetadata(h); end
setmetadata!(backend::InMemoryStorage, i::Integer; kwargs...) = open(backend, i) do h; setmetadata!(h; kwargs...); end
ensure_output!(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; ensure_output!(h, name); end
store_output!(backend::InMemoryStorage, i::Integer, name::Symbol, x) = (open(backend, i) do h; store_output!(h, name, x); end; backend)
get_output(backend::InMemoryStorage, i::Integer, name::Symbol, j::Integer) = open(backend, i) do h; get_output(h, name, j); end
get_outputs(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; get_outputs(h, name) end
output_length(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; output_length(h, name); end
output_names(backend::InMemoryStorage, i::Integer) = open(backend, i) do h; output_names(h); end
has_output(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; has_output(h, name); end
empty_output!(backend::InMemoryStorage, i::Integer, name::Symbol) = (open(backend, i) do h; empty_output!(h, name); end; backend)

Base.empty!(backend::InMemoryStorage, i::Integer) = (open(backend, i) do h; empty!(h); end; backend)
Base.empty!(backend::InMemoryStorage) = (empty!(backend.inputs); empty!(backend.metadata); empty!(backend.outputs); backend)

function Base.copy!(dest::InMemoryStorage, src::InMemoryStorage, i::Integer)
    dest.inputs[i] = deepcopy(src.inputs[i])
    dest.metadata[i] = deepcopy(src.metadata[i])
    dest.outputs[i] = deepcopy(src.outputs[i])
end

"""
    InMemoryStorageHandle <: StorageHandle

Handle for [`InMemoryStorage`](@ref) for a specific simulation. Output operations forward
directly to the backend; the ephemeral scratch namespace lives here (owned by this handle,
not indexed by simulation) and is cleared on close.
"""
mutable struct InMemoryStorageHandle <: StorageHandle
    backend::InMemoryStorage # Storage backend
    index::Int  # Simulation ID this handle owns
    isopen::Bool # Handle status
    scratch::Dict{Symbol, Any}  # Scratch storage for the current simulation
end

"""
    open(backend::InMemoryStorage, index::Integer; kwargs...)

Open a handle for the `index`-th simulation. The handle owns the scratch namespace for
that specific simulation.
"""
function Base.open(backend::InMemoryStorage, index::Integer; kwargs...)
    handle = InMemoryStorageHandle(backend, Int(index), true, Dict{Symbol, Any}())
    finalizer(close, handle)
    return handle
end

function Base.close(handle::InMemoryStorageHandle)
    empty!(handle.scratch)
    handle.isopen = false
end

Base.isopen(handle::InMemoryStorageHandle) = handle.isopen

# Generic in-memory scratch implementation shared by all handle types.
# Scratch is per-simulation, owned by the handle itself (not indexed).
_ensure_scratch!(scratch::Dict{Symbol, Any}, name::Symbol) = begin
    haskey(scratch, name) || (scratch[name] = Any[])
    return scratch[name]
end

ensure_scratch!(h::StorageHandle, name::Symbol) = _ensure_scratch!(h.scratch, name)
store_scratch!(h::StorageHandle, name::Symbol, x) = push!(_ensure_scratch!(h.scratch, name), x)
get_scratch(h::StorageHandle, name::Symbol, j::Integer) = h.scratch[name][j]
scratch_length(h::StorageHandle, name::Symbol) = haskey(h.scratch, name) ? length(h.scratch[name]) : 0
scratch_names(h::StorageHandle) = collect(keys(h.scratch))
has_scratch(h::StorageHandle, name::Symbol) = haskey(h.scratch, name)
empty_scratch!(h::StorageHandle, name::Symbol) = haskey(h.scratch, name) && empty!(h.scratch[name])

# --- handle-based operations (fast path) ---

num_simulations(handle::InMemoryStorageHandle) = length(handle.backend.inputs)

function allocate!(handle::InMemoryStorageHandle, input; metadata...)
    backend = handle.backend
    push!(backend.inputs, input)
    push!(backend.metadata, OrderedDict{Symbol,Any}(metadata))
    push!(backend.outputs, OrderedDict{Symbol,Vector{Any}}())
    return length(backend.inputs)
end

getinputs(handle::InMemoryStorageHandle) = handle.backend.inputs[handle.index]
setinputs!(handle::InMemoryStorageHandle, x) = (handle.backend.inputs[handle.index] = x; handle)
getmetadata(handle::InMemoryStorageHandle) = handle.backend.metadata[handle.index]
setmetadata!(handle::InMemoryStorageHandle; kwargs...) = [setindex!(handle.backend.metadata[handle.index], kv...) for kv in kwargs]

function ensure_output!(handle::InMemoryStorageHandle, name::Symbol)
    dict = handle.backend.outputs[handle.index]
    haskey(dict, name) || (dict[name] = valtype(dict)())
    return dict[name]
end
store_output!(handle::InMemoryStorageHandle, name::Symbol, x) = push!(ensure_output!(handle, name), x)
get_output(handle::InMemoryStorageHandle, name::Symbol, j::Integer) = handle.backend.outputs[handle.index][name][j]
get_outputs(handle::InMemoryStorageHandle, name::Symbol) = collect(handle.backend.outputs[handle.index][name])
output_length(handle::InMemoryStorageHandle, name::Symbol) = haskey(handle.backend.outputs[handle.index], name) ? length(handle.backend.outputs[handle.index][name]) : 0
output_names(handle::InMemoryStorageHandle) = collect(keys(handle.backend.outputs[handle.index]))
has_output(handle::InMemoryStorageHandle, name::Symbol) = haskey(handle.backend.outputs[handle.index], name)
empty_output!(handle::InMemoryStorageHandle, name::Symbol) = (haskey(handle.backend.outputs[handle.index], name) && empty!(handle.backend.outputs[handle.index][name]); handle)

function Base.empty!(handle::InMemoryStorageHandle)
    empty!(handle.backend.outputs[handle.index])
    return handle
end

############################################################
# On-disk backend entrypoint
############################################################

const SUPPORTED_DISK_FORMATS = [format"JLD2"]

"""
    DiskStorageBackend(::Type{DataFormat{format}}, path::AbstractString, args...; kwargs...) where {format}

Construct a disk-backed `StorageBackend` whose backend persists each simulation to disk at the given `path`.
Requires a supported file I/O backend to be loaded, e.g. `JLD2`.
"""
DiskStorageBackend(::Type{DataFormat{format}}, path::AbstractString, args...; kwargs...) where {format} = error(
    "No disk storage backend loaded for $format. Load the corresponding package for one of the supported formats $SUPPORTED_DISK_FORMATS to enable this backend.",
)
