module SimulationBasedInferenceJLD2Ext

using SimulationBasedInference
using SimulationBasedInference: StorageBackend, StorageHandle, SimulationDataSet,
    num_simulations, allocate!, getinputs, setinputs!, getmetadata, setmetadata!,
    ensure_output!, store_output!, get_output, get_outputs, output_length, output_names, has_output, empty_output!,
    ensure_scratch!, store_scratch!, get_scratch, scratch_length, scratch_names, has_scratch, empty_scratch!

using FileIO
using JLD2

"""
    JLD2Storage <: StorageBackend

Disk-backed storage backend that persists each simulation as a group in a JLD2 file.

Outputs are always persisted. Scratch (transient working) series are governed by
`scratch_in_memory`: when `true` (default) they live in the handle's in-memory dict and are
discarded on close; when `false` they are written to disk under `simulations/<i>/scratch`.

Layout per simulation `i`:

    simulations/<i>/input
    simulations/<i>/metadata
    simulations/<i>/outputs/<name>/<j>     # j = 1-based element index
    simulations/<i>/scratch/<name>/<j>     # only when scratch_in_memory == false
"""
mutable struct JLD2Storage <: StorageBackend
    path::String
    num_simulations::Int
    scratch_in_memory::Bool
end

function _num_simulations(path::AbstractString)
    isfile(path) || return 0
    return JLD2.jldopen(path, "r") do file
        haskey(file, "simulations") ? length(keys(file["simulations"])) : 0
    end
end

function JLD2Storage(path::AbstractString; overwrite::Bool=false, scratch_in_memory::Bool=true)
    if !isfile(path) || overwrite
        JLD2.jldopen(path, "w") do _ end  # create/truncate
        n = 0
    else
        n = _num_simulations(path)
    end
    return JLD2Storage(String(path), n, scratch_in_memory)
end

"""
    OnDiskSimulationDataSet(file::File{format"JLD2"}; overwrite=false, scratch_in_memory=true)

Construct a `SimulationDataSet` backed by a [`JLD2Storage`](@ref) at `file`.
"""
SimulationBasedInference.OnDiskSimulationDataSet(file::File{format"JLD2"}; kwargs...) =
    SimulationDataSet(JLD2Storage(file.filename; kwargs...))

# --- handle ---

"""
    JLD2StorageHandle <: StorageHandle

Handle for [`JLD2Storage`](@ref). Holds the JLD2 file open for the duration of a batch of
operations and owns the in-memory scratch namespace (used unless `scratch_in_memory` is
`false`, in which case scratch is written to disk).
"""
mutable struct JLD2StorageHandle <: StorageHandle
    backend::JLD2Storage
    file::Union{JLD2.JLDFile, Nothing}
    isopen::Bool
    scratch::Dict{Symbol,Any}
end

function Base.open(backend::JLD2Storage, mode::AbstractString = "a+")
    file = JLD2.jldopen(backend.path, mode)
    handle = JLD2StorageHandle(backend, file, true, Dict{Symbol,Any}())
    finalizer(close, handle)
    return handle
end

function Base.close(handle::JLD2StorageHandle)
    empty!(handle.scratch)
    if handle.isopen && handle.file !== nothing
        JLD2.close(handle.file)
        handle.file = nothing
        handle.isopen = false
    end
    return nothing
end

# --- shared file helpers (operate on an open JLD2 file) ---
_fkey(i, group, name) = "simulations/$i/$group/$name"

function _file_store!(file, i::Integer, group::Symbol, name::Symbol, x)
    key = _fkey(i, group, name)
    n = haskey(file, key) ? length(keys(file[key])) : 0
    file["$key/$(n + 1)"] = x
    return nothing
end

_file_get(file, i::Integer, group::Symbol, name::Symbol, j::Integer) = file["$(_fkey(i, group, name))/$j"]

_file_length(file, i::Integer, group::Symbol, name::Symbol) = (k = _fkey(i, group, name); haskey(file, k) ? length(keys(file[k])) : 0)

function _file_names(file, i::Integer, group::Symbol)
    g = "simulations/$i/$group"
    return haskey(file, g) ? sort!(Symbol.(collect(keys(file[g])))) : Symbol[]
end

_file_has(file, i::Integer, group::Symbol, name::Symbol) = haskey(file, _fkey(i, group, name))

_file_clear!(file, i::Integer, group::Symbol, name::Symbol) = (k = _fkey(i, group, name); haskey(file, k) && delete!(file, k); nothing)

_file_clear_group!(file, i::Integer, group::Symbol) = (g = "simulations/$i/$group"; haskey(file, g) && delete!(file, g); nothing)

function _read_metadata(file, i::Integer)
    key = "simulations/$i/metadata"
    group = file[key]
    return Dict((Symbol(k) => group[k] for k in keys(group)))
end

# --- handle-based operations (fast path - uses the open file) ---

SimulationBasedInference.num_simulations(h::JLD2StorageHandle) = h.backend.num_simulations

function SimulationBasedInference.allocate!(h::JLD2StorageHandle, input; metadata...)
    i = h.backend.num_simulations + 1
    h.file["simulations/$i/input"] = input
    for kv in metadata
        h.file["simulations/$i/metadata/$(first(kv))"] = last(kv)
    end
    h.backend.num_simulations = i
    return i
end

SimulationBasedInference.getinputs(h::JLD2StorageHandle, i::Integer) = h.file["simulations/$i/input"]

function SimulationBasedInference.setinputs!(h::JLD2StorageHandle, i::Integer, x)
    key = "simulations/$i/input"
    haskey(h.file, key) && delete!(h.file, key)  # JLD2 datasets are write-once
    h.file[key] = x
    return h
end

SimulationBasedInference.getmetadata(h::JLD2StorageHandle, i::Integer) = _read_metadata(h.file, i)

function SimulationBasedInferenceJLD2Ext.setmetadata!(h::JLD2StorageHandle, i::Integer; kwargs...)
    key = "simulations/$i/metadata"
    old_metadata = Dict{Symbol, Any}()
    if haskey(h.file, key)
        copy!(old_metadata, _read_metadata(h.file, i))
        delete!(h.file, key)  # JLD2 datasets are write-once
    end
    for kv in merge(old_metadata, kwargs)
        h.file["$key/$(first(kv))"] = last(kv)
    end
    return h
end

# output series (always on disk)
SimulationBasedInference.ensure_output!(::JLD2StorageHandle, ::Integer, ::Symbol) = nothing
SimulationBasedInference.store_output!(h::JLD2StorageHandle, i::Integer, name::Symbol, x) = _file_store!(h.file, i, :outputs, name, x)
SimulationBasedInference.get_output(h::JLD2StorageHandle, i::Integer, name::Symbol, j::Integer) = _file_get(h.file, i, :outputs, name, j)
SimulationBasedInference.get_outputs(h::JLD2StorageHandle, i::Integer, name::Symbol) = [_file_get(h.file, i, :outputs, name, j) for j in 1:_file_length(h.file, i, :outputs, name)]
SimulationBasedInference.output_length(h::JLD2StorageHandle, i::Integer, name::Symbol) = _file_length(h.file, i, :outputs, name)
SimulationBasedInference.output_names(h::JLD2StorageHandle, i::Integer) = _file_names(h.file, i, :outputs)
SimulationBasedInference.has_output(h::JLD2StorageHandle, i::Integer, name::Symbol) = _file_has(h.file, i, :outputs, name)
SimulationBasedInference.empty_output!(h::JLD2StorageHandle, i::Integer, name::Symbol) = _file_clear!(h.file, i, :outputs, name)

# scratch series (in-memory by default, on disk when scratch_in_memory == false)
function SimulationBasedInference.ensure_scratch!(h::JLD2StorageHandle, ::Integer, name::Symbol)
    h.backend.scratch_in_memory && _ensure_scratch!(h.scratch, name)
    return nothing
end
function SimulationBasedInference.store_scratch!(h::JLD2StorageHandle, i::Integer, name::Symbol, x)
    if h.backend.scratch_in_memory
        push!(_ensure_scratch!(h.scratch, name), x)
    else
        _file_store!(h.file, i, :scratch, name, x)
    end
    return nothing
end
SimulationBasedInference.get_scratch(h::JLD2StorageHandle, i::Integer, name::Symbol, j::Integer) =
    h.backend.scratch_in_memory ? h.scratch[name][j] : _file_get(h.file, i, :scratch, name, j)
SimulationBasedInference.scratch_length(h::JLD2StorageHandle, i::Integer, name::Symbol) =
    h.backend.scratch_in_memory ? (haskey(h.scratch, name) ? length(h.scratch[name]) : 0) : _file_length(h.file, i, :scratch, name)
SimulationBasedInference.scratch_names(h::JLD2StorageHandle, i::Integer) =
    h.backend.scratch_in_memory ? collect(keys(h.scratch)) : _file_names(h.file, i, :scratch)
SimulationBasedInference.has_scratch(h::JLD2StorageHandle, i::Integer, name::Symbol) =
    h.backend.scratch_in_memory ? haskey(h.scratch, name) : _file_has(h.file, i, :scratch, name)
function SimulationBasedInference.empty_scratch!(h::JLD2StorageHandle, i::Integer, name::Symbol)
    if h.backend.scratch_in_memory
        haskey(h.scratch, name) && empty!(h.scratch[name])
    else
        _file_clear!(h.file, i, :scratch, name)
    end
    return nothing
end

function Base.empty!(h::JLD2StorageHandle, i::Integer)
    _file_clear_group!(h.file, i, :outputs)
    h.backend.scratch_in_memory || _file_clear_group!(h.file, i, :scratch)
    return nothing
end

# --- backend operations (slow path - auto-open/close) ---

SimulationBasedInference.num_simulations(b::JLD2Storage) = b.num_simulations
SimulationBasedInference.allocate!(b::JLD2Storage, input; metadata...) = open(b) do h; allocate!(h, input; metadata...); end
SimulationBasedInference.getinputs(b::JLD2Storage, i::Integer) = open(b) do h; getinputs(h, i); end
SimulationBasedInference.setinputs!(b::JLD2Storage, i::Integer, x) = open(b) do h; setinputs!(h, i, x); end
SimulationBasedInference.getmetadata(b::JLD2Storage, i::Integer) = open(b) do h; getmetadata(h, i); end
SimulationBasedInference.setmetadata!(b::JLD2Storage, i::Integer; kwargs...) = open(b) do h; setmetadata!(h, i; kwargs...); end
SimulationBasedInference.ensure_output!(::JLD2Storage, ::Integer, ::Symbol) = nothing
SimulationBasedInference.store_output!(b::JLD2Storage, i::Integer, name::Symbol, x) = open(b) do h; store_output!(h, i, name, x); end
SimulationBasedInference.get_output(b::JLD2Storage, i::Integer, name::Symbol, j::Integer) = open(b) do h; get_output(h, i, name, j); end
SimulationBasedInference.get_outputs(b::JLD2Storage, i::Integer, name::Symbol) = open(b) do h; get_outputs(h, i, name); end
SimulationBasedInference.output_length(b::JLD2Storage, i::Integer, name::Symbol) = open(b) do h; output_length(h, i, name); end
SimulationBasedInference.output_names(b::JLD2Storage, i::Integer) = open(b) do h; output_names(h, i); end
SimulationBasedInference.has_output(b::JLD2Storage, i::Integer, name::Symbol) = open(b) do h; has_output(h, i, name); end
SimulationBasedInference.empty_output!(b::JLD2Storage, i::Integer, name::Symbol) = open(b) do h; empty_output!(h, i, name); end

Base.empty!(b::JLD2Storage, i::Integer) = open(b) do h; empty!(h, i) end
function Base.empty!(b::JLD2Storage)
    b.num_simulations = 0
    JLD2.jldopen(b.path, "w") do _ end  # truncate
    return nothing
end

end
