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

Abstract base type for the object returned by `open(backend::StorageBackend)`. A handle holds
the active connection to a backend (e.g. an open file) for the duration of a batch of
operations and owns the **scratch** namespace. Concrete handle types must provide a
`scratch::Dict{Symbol,Any}` field; scratch is cleared when the handle is closed.

Use with the do-block form for guaranteed cleanup:

    open(backend) do h
        # use handle h
    end  # handle closed (and scratch cleared) automatically
"""
abstract type StorageHandle end

Base.open(backend::StorageBackend) = error("open not defined for $(typeof(backend))")

# Generic in-memory scratch implementation shared by all handle types.
# Scratch is keyed by name within a (per-simulation) handle, so the simulation index is
# accepted for interface uniformity but ignored by the in-memory implementation.
_scratch_ensure!(d::AbstractDict, name::Symbol) = (haskey(d, name) || (d[name] = Any[]); d[name])
ensure_scratch!(h::StorageHandle, ::Integer, name::Symbol) = (_scratch_ensure!(h.scratch, name); h)
store_scratch!(h::StorageHandle, ::Integer, name::Symbol, x) = (push!(_scratch_ensure!(h.scratch, name), x); h)
get_scratch(h::StorageHandle, ::Integer, name::Symbol, j::Integer) = h.scratch[name][j]
scratch_length(h::StorageHandle, ::Integer, name::Symbol) = haskey(h.scratch, name) ? length(h.scratch[name]) : 0
scratch_names(h::StorageHandle, ::Integer) = collect(keys(h.scratch))
has_scratch(h::StorageHandle, ::Integer, name::Symbol) = haskey(h.scratch, name)
empty_scratch!(h::StorageHandle, ::Integer, name::Symbol) = (haskey(h.scratch, name) && empty!(h.scratch[name]); h)

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

InMemoryStorage{I,O,M}() where {I,O,M} = InMemoryStorage{I,O,M}(
    I[],
    OrderedDict{Symbol,M}[],
    OrderedDict{Symbol,Vector{O}}[],
)

InMemoryStorage() = InMemoryStorage{Any,Any,Any}()

"""
    InMemoryStorageHandle <: StorageHandle

Handle for [`InMemoryStorage`](@ref). Output operations forward directly to the backend; the
ephemeral scratch namespace lives here and is cleared on close.
"""
mutable struct InMemoryStorageHandle <: StorageHandle
    backend::InMemoryStorage
    scratch::Dict{Symbol,Any}
end

function Base.open(backend::InMemoryStorage, mode=nothing)
    handle = InMemoryStorageHandle(backend, Dict{Symbol,Any}())
    finalizer(close, handle)
    return handle
end

Base.close(handle::InMemoryStorageHandle) = empty!(handle.scratch)

# --- handle-based operations (fast path) ---

num_simulations(handle::InMemoryStorageHandle) = length(handle.backend.inputs)

function allocate!(handle::InMemoryStorageHandle, input; metadata...)
    backend = handle.backend
    push!(backend.inputs, input)
    push!(backend.metadata, OrderedDict{Symbol,Any}(metadata))
    push!(backend.outputs, OrderedDict{Symbol,Vector{Any}}())
    return length(backend.inputs)
end

getinputs(handle::InMemoryStorageHandle, i::Integer) = handle.backend.inputs[i]
setinputs!(handle::InMemoryStorageHandle, i::Integer, x) = (handle.backend.inputs[i] = x; handle)
getmetadata(handle::InMemoryStorageHandle, i::Integer) = handle.backend.metadata[i]
setmetadata!(handle::InMemoryStorageHandle, i::Integer; kwargs...) = [setindex!(handle.backend.metadata[i], kv...) for kv in kwargs]

function ensure_output!(handle::InMemoryStorageHandle, i::Integer, name::Symbol)
    dict = handle.backend.outputs[i]
    haskey(dict, name) || (dict[name] = valtype(dict)())
    return dict[name]
end
store_output!(handle::InMemoryStorageHandle, i::Integer, name::Symbol, x) = push!(ensure_output!(handle, i, name), x)
get_output(handle::InMemoryStorageHandle, i::Integer, name::Symbol, j::Integer) = handle.backend.outputs[i][name][j]
get_outputs(handle::InMemoryStorageHandle, i::Integer, name::Symbol) = collect(handle.backend.outputs[i][name])
output_length(handle::InMemoryStorageHandle, i::Integer, name::Symbol) = haskey(handle.backend.outputs[i], name) ? length(handle.backend.outputs[i][name]) : 0
output_names(handle::InMemoryStorageHandle, i::Integer) = collect(keys(handle.backend.outputs[i]))
has_output(handle::InMemoryStorageHandle, i::Integer, name::Symbol) = haskey(handle.backend.outputs[i], name)
empty_output!(handle::InMemoryStorageHandle, i::Integer, name::Symbol) = (haskey(handle.backend.outputs[i], name) && empty!(handle.backend.outputs[i][name]); handle)

function Base.empty!(handle::InMemoryStorageHandle, i::Integer)
    empty!(handle.backend.outputs[i])
    return handle
end

# --- backend operations (slow path - auto-open/close for backward compatibility) ---

num_simulations(backend::InMemoryStorage) = length(backend.inputs)
allocate!(backend::InMemoryStorage, input; metadata...) = open(backend) do h; allocate!(h, input; metadata...); end
getinputs(backend::InMemoryStorage, i::Integer) = open(backend) do h; getinputs(h, i); end
setinputs!(backend::InMemoryStorage, i::Integer, x) = (open(backend) do h; setinputs!(h, i, x); end; backend)
getmetadata(backend::InMemoryStorage, i::Integer) = open(backend) do h; getmetadata(h, i); end
setmetadata!(bacend::InMemoryStorage, i::Integer; kwargs...) = open(backend) do h; setmetdata!(h, i; kwargs...); end
ensure_output!(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend) do h; ensure_output!(h, i, name); end
store_output!(backend::InMemoryStorage, i::Integer, name::Symbol, x) = (open(backend) do h; store_output!(h, i, name, x); end; backend)
get_output(backend::InMemoryStorage, i::Integer, name::Symbol, j::Integer) = open(backend) do h; get_output(h, i, name, j); end
get_outputs(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend) do h; get_outputs(h, i, name) end
output_length(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend) do h; output_length(h, i, name); end
output_names(backend::InMemoryStorage, i::Integer) = open(backend) do h; output_names(h, i); end
has_output(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend) do h; has_output(h, i, name); end
empty_output!(backend::InMemoryStorage, i::Integer, name::Symbol) = (open(backend) do h; empty_output!(h, i, name); end; backend)

Base.empty!(backend::InMemoryStorage, i::Integer) = (empty!(backend.outputs[i]); backend)
function Base.empty!(backend::InMemoryStorage)
    empty!(backend.inputs)
    empty!(backend.metadata)
    empty!(backend.outputs)
    return backend
end
