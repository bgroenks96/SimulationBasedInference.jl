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

Abstract base type for the object returned by `open(backend::StorageBackend, sim_id::Integer)`. A handle holds
the active connection to a backend (e.g. an open file or in-memory view) for the duration of a batch of
operations on a specific simulation and owns the **scratch** namespace for that simulation.
Concrete handle types must provide a `scratch::Dict{Symbol,Any}` field; scratch is cleared when the handle is closed.

Use with the do-block form for guaranteed cleanup:

    open(backend, sim_id) do h
        # use handle h for simulation sim_id
    end  # handle closed (and scratch cleared) automatically
"""
abstract type StorageHandle end

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

Handle for [`InMemoryStorage`](@ref) for a specific simulation. Output operations forward
directly to the backend; the ephemeral scratch namespace lives here (owned by this handle,
not indexed by simulation) and is cleared on close.
"""
mutable struct InMemoryStorageHandle <: StorageHandle
    backend::InMemoryStorage
    sim_id::Int  # Simulation ID this handle owns
    scratch::Dict{Symbol, Any}  # Scratch for THIS simulation only
end

"""
    open(backend::InMemoryStorage, sim_id::Integer; args...)

Open a handle for the `sim_id`-th simulation. The handle owns the scratch namespace for
that specific simulation.
"""
function Base.open(backend::InMemoryStorage, sim_id::Integer; args...)
    handle = InMemoryStorageHandle(backend, Int(sim_id), Dict{Symbol, Any}())
    finalizer(close, handle)
    return handle
end

Base.close(handle::InMemoryStorageHandle) = empty!(handle.scratch)

# Generic in-memory scratch implementation shared by all handle types.
# Scratch is per-simulation, owned by the handle itself (not indexed).
_ensure_scratch!(scratch::Dict{Symbol, Any}, name::Symbol) = begin
    haskey(scratch, name) || (scratch[name] = Any[])
    return scratch[name]
end

ensure_scratch!(h::StorageHandle, ::Integer, name::Symbol) = _ensure_scratch!(h.scratch, name)
store_scratch!(h::StorageHandle, ::Integer, name::Symbol, x) = push!(_ensure_scratch!(h.scratch, name), x)
get_scratch(h::StorageHandle, ::Integer, name::Symbol, j::Integer) = h.scratch[name][j]
scratch_length(h::StorageHandle, ::Integer, name::Symbol) = haskey(h.scratch, name) ? length(h.scratch[name]) : 0
scratch_names(h::StorageHandle, ::Integer) = collect(keys(h.scratch))
has_scratch(h::StorageHandle, ::Integer, name::Symbol) = haskey(h.scratch, name)
empty_scratch!(h::StorageHandle, ::Integer, name::Symbol) = haskey(h.scratch, name) && empty!(h.scratch[name])

# --- handle-based operations (fast path) ---

num_simulations(handle::InMemoryStorageHandle) = length(handle.backend.inputs)

function allocate!(handle::InMemoryStorageHandle, input; metadata...)
    backend = handle.backend
    push!(backend.inputs, input)
    push!(backend.metadata, OrderedDict{Symbol,Any}(metadata))
    push!(backend.outputs, OrderedDict{Symbol,Vector{Any}}())
    return length(backend.inputs)
end

getinputs(handle::InMemoryStorageHandle, ::Integer) = handle.backend.inputs[handle.sim_id]
setinputs!(handle::InMemoryStorageHandle, ::Integer, x) = (handle.backend.inputs[handle.sim_id] = x; handle)
getmetadata(handle::InMemoryStorageHandle, ::Integer) = handle.backend.metadata[handle.sim_id]
setmetadata!(handle::InMemoryStorageHandle, ::Integer; kwargs...) = [setindex!(handle.backend.metadata[handle.sim_id], kv...) for kv in kwargs]

function ensure_output!(handle::InMemoryStorageHandle, ::Integer, name::Symbol)
    dict = handle.backend.outputs[handle.sim_id]
    haskey(dict, name) || (dict[name] = valtype(dict)())
    return dict[name]
end
store_output!(handle::InMemoryStorageHandle, ::Integer, name::Symbol, x) = push!(ensure_output!(handle, handle.sim_id, name), x)
get_output(handle::InMemoryStorageHandle, ::Integer, name::Symbol, j::Integer) = handle.backend.outputs[handle.sim_id][name][j]
get_outputs(handle::InMemoryStorageHandle, ::Integer, name::Symbol) = collect(handle.backend.outputs[handle.sim_id][name])
output_length(handle::InMemoryStorageHandle, ::Integer, name::Symbol) = haskey(handle.backend.outputs[handle.sim_id], name) ? length(handle.backend.outputs[handle.sim_id][name]) : 0
output_names(handle::InMemoryStorageHandle, ::Integer) = collect(keys(handle.backend.outputs[handle.sim_id]))
has_output(handle::InMemoryStorageHandle, ::Integer, name::Symbol) = haskey(handle.backend.outputs[handle.sim_id], name)
empty_output!(handle::InMemoryStorageHandle, ::Integer, name::Symbol) = (haskey(handle.backend.outputs[handle.sim_id], name) && empty!(handle.backend.outputs[handle.sim_id][name]); handle)

function Base.empty!(handle::InMemoryStorageHandle, ::Integer)
    empty!(handle.backend.outputs[handle.sim_id])
    return handle
end

# --- backend operations (slow path - auto-open/close per simulation) ---

num_simulations(backend::InMemoryStorage) = length(backend.inputs)
allocate!(backend::InMemoryStorage, input; metadata...) = open(backend, length(backend.inputs) + 1) do h; allocate!(h, input; metadata...); end
getinputs(backend::InMemoryStorage, i::Integer) = open(backend, i) do h; getinputs(h, i); end
setinputs!(backend::InMemoryStorage, i::Integer, x) = (open(backend, i) do h; setinputs!(h, i, x); end; backend)
getmetadata(backend::InMemoryStorage, i::Integer) = open(backend, i) do h; getmetadata(h, i); end
setmetadata!(backend::InMemoryStorage, i::Integer; kwargs...) = open(backend, i) do h; setmetadata!(h, i; kwargs...); end
ensure_output!(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; ensure_output!(h, i, name); end
store_output!(backend::InMemoryStorage, i::Integer, name::Symbol, x) = (open(backend, i) do h; store_output!(h, i, name, x); end; backend)
get_output(backend::InMemoryStorage, i::Integer, name::Symbol, j::Integer) = open(backend, i) do h; get_output(h, i, name, j); end
get_outputs(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; get_outputs(h, i, name) end
output_length(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; output_length(h, i, name); end
output_names(backend::InMemoryStorage, i::Integer) = open(backend, i) do h; output_names(h, i); end
has_output(backend::InMemoryStorage, i::Integer, name::Symbol) = open(backend, i) do h; has_output(h, i, name); end
empty_output!(backend::InMemoryStorage, i::Integer, name::Symbol) = (open(backend, i) do h; empty_output!(h, i, name); end; backend)

Base.empty!(backend::InMemoryStorage, i::Integer) = (open(backend, i) do h; empty!(h, i); end; backend)
function Base.empty!(backend::InMemoryStorage)
    empty!(backend.inputs)
    empty!(backend.metadata)
    empty!(backend.outputs)
    return backend
end
