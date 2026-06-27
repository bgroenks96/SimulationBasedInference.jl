"""
    StorageBackend

Abstract type for the storage backend that owns all simulation data. There is exactly one
backend per [`SimulationDataSet`](@ref); the higher-level types ([`SimulationData`](@ref),
[`DataBuffer`](@ref))s are lightweight views that route all reads and writes through it. The
backend decides where the bytes actually live — in memory ([`InMemoryStorage`](@ref)) or
out-of-core (e.g. the JLD2 backend provided by the `SimulationBasedInferenceJLD2Ext`
extension).

Each simulation has an input, a metadata dictionary, a set of named **output** series, and a
set of named **scratch** series (transient working buffers).
"""
abstract type StorageBackend end

"""
    InMemoryStorage{inputType,outputType,scratchType,metadataType} <: StorageBackend

In-memory storage backend. Simulation inputs, output series elements, scratch series
elements, and metadata values are strongly typed as `inputType`, `outputType`,
`scratchType`, and `metadataType` respectively. The default constructor `InMemoryStorage()`
uses `AbstractArray` for the inputs/outputs/scratch (matching the common case where inputs
are parameter vectors and observable outputs are arrays) and `Any` for metadata values.
"""
mutable struct InMemoryStorage{inputType,outputType,scratchType,metadataType} <: StorageBackend
    inputs::Vector{inputType}
    metadata::Vector{OrderedDict{Symbol,metadataType}}
    outputs::Vector{OrderedDict{Symbol,Vector{outputType}}}
    scratch::Vector{OrderedDict{Symbol,Vector{scratchType}}}
end

InMemoryStorage{I,O,S,M}() where {I,O,S,M} = InMemoryStorage{I,O,S,M}(
    I[],
    OrderedDict{Symbol,M}[],
    OrderedDict{Symbol,Vector{O}}[],
    OrderedDict{Symbol,Vector{S}}[],
)

InMemoryStorage() = InMemoryStorage{Any,Any,Any,Any}()

num_simulations(backend::InMemoryStorage) = length(backend.inputs)

function allocate!(backend::InMemoryStorage{I,O,S,M}, input; metadata...) where {I,O,S,M}
    push!(backend.inputs, input)
    push!(backend.metadata, OrderedDict{Symbol,M}(metadata))
    push!(backend.outputs, OrderedDict{Symbol,Vector{O}}())
    push!(backend.scratch, OrderedDict{Symbol,Vector{S}}())
    return length(backend.inputs)
end

getinputs(backend::InMemoryStorage, i::Integer) = backend.inputs[i]
setinputs!(backend::InMemoryStorage, i::Integer, x) = (backend.inputs[i] = x; backend)
getmetadata(backend::InMemoryStorage, i::Integer) = backend.metadata[i]

# --- output series ---
function ensure_output!(backend::InMemoryStorage, i::Integer, name::Symbol)
    dict = backend.outputs[i]
    if !haskey(dict, name)
        dict[name] = valtype(dict)[]
    end
    return dict[name]
end
store_output!(backend::InMemoryStorage, i::Integer, name::Symbol, x) = (push!(ensure_output!(backend, i, name), x); backend)
get_output(backend::InMemoryStorage, i::Integer, name::Symbol, j::Integer) = backend.outputs[i][name][j]
output_length(backend::InMemoryStorage, i::Integer, name::Symbol) = haskey(backend.outputs[i], name) ? length(backend.outputs[i][name]) : 0
output_names(backend::InMemoryStorage, i::Integer) = collect(keys(backend.outputs[i]))
has_output(backend::InMemoryStorage, i::Integer, name::Symbol) = haskey(backend.outputs[i], name)
empty_output!(backend::InMemoryStorage, i::Integer, name::Symbol) = haskey(backend.outputs[i], name) && empty!(backend.outputs[i][name])

# --- scratch series ---
function ensure_scratch!(backend::InMemoryStorage, i::Integer, name::Symbol)
    dict = backend.scratch[i]
    if !haskey(dict, name)
        dict[name] = valtype(dict)[]
    end
    return dict[name]
end
store_scratch!(backend::InMemoryStorage, i::Integer, name::Symbol, x) = (push!(ensure_scratch!(backend, i, name), x); backend)
get_scratch(backend::InMemoryStorage, i::Integer, name::Symbol, j::Integer) = backend.scratch[i][name][j]
scratch_length(backend::InMemoryStorage, i::Integer, name::Symbol) = haskey(backend.scratch[i], name) ? length(backend.scratch[i][name]) : 0
scratch_names(backend::InMemoryStorage, i::Integer) = collect(keys(backend.scratch[i]))
has_scratch(backend::InMemoryStorage, i::Integer, name::Symbol) = haskey(backend.scratch[i], name)
empty_scratch!(backend::InMemoryStorage, i::Integer, name::Symbol) = haskey(backend.scratch[i], name) && empty!(backend.scratch[i][name])

# --- whole-simulation / whole-backend ---
function Base.empty!(backend::InMemoryStorage, i::Integer)
    empty!(backend.outputs[i])
    empty!(backend.scratch[i])
    return backend
end

function Base.empty!(backend::InMemoryStorage)
    empty!(backend.inputs)
    empty!(backend.metadata)
    empty!(backend.outputs)
    empty!(backend.scratch)
    return backend
end
