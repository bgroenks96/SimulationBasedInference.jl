module SimulationBasedInferenceJLD2Ext

using SimulationBasedInference
using SimulationBasedInference: StorageBackend, StorageHandle, SimulationDataSet,
    num_simulations, allocate!, getinputs, setinputs!, getmetadata, setmetadata!,
    ensure_output!, store_output!, get_output, get_outputs, output_length, output_names, has_output, empty_output!,
    ensure_scratch!, store_scratch!, get_scratch, scratch_length, scratch_names, has_scratch, empty_scratch!

using FileIO, JLD2
using Printf
using Dates

"""
    ParallelJLD2Storage <: StorageBackend

Disk-backed storage backend that persists each simulation as a separate JLD2 file in a
directory structure. This enables parallel access to simulations without file locking conflicts.

Outputs are always persisted. Scratch (transient working) series are governed by
`scratch_in_memory`: when `true` (default) they live in the handle's in-memory dict and are
discarded on close; when `false` they are written to disk in the JLD2 file.

Directory layout:
    <path>/
        simulation_0001.jld2   # simulation 1
        simulation_0002.jld2   # simulation 2
            ...
"""
mutable struct ParallelJLD2Storage <: StorageBackend
    path::String
    num_simulations::Int
    scratch_in_memory::Bool
end

function _num_simulations(path::AbstractString)
    isdir(path) || return 0
    files = filter(endswith(".jld2"), readdir(path))
    return length(files)
end

function ParallelJLD2Storage(path::AbstractString; overwrite::Bool=false, scratch_in_memory::Bool=true)    
    if isdir(path) && overwrite
        # Remove existing simulations
        rm(path; recursive=true)
    end
    # Create directory structure
    mkpath(path)
    n = _num_simulations(path)
    return ParallelJLD2Storage(String(path), n, scratch_in_memory)
end

"""
    OnDiskSimulationDataSet(::Type{DataFormat{:JLD2}}, dir::AbstractString; kwargs...)

Construct a `SimulationDataSet` backed by a [`ParallelJLD2Storage`](@ref) at `dir`.
The directory will contain individual JLD2 files for each simulation.
"""
SimulationBasedInference.OnDiskSimulationDataSet(::Type{DataFormat{:JLD2}}, dir::AbstractString; kwargs...) =
    SimulationDataSet(ParallelJLD2Storage(dir; kwargs...))

# --- handle ---

"""
    ParallelJLD2StorageHandle <: StorageHandle

Handle for a single simulation's JLD2 file in [`ParallelJLD2Storage`](@ref). Each handle
owns the connection to one `simulations/<i>.jld2` file and the per-simulation scratch
namespace. Handles are created per-simulation via `open(backend, sim_id)`.
"""
mutable struct ParallelJLD2StorageHandle <: StorageHandle
    backend::ParallelJLD2Storage
    sim_id::Int
    file::Union{JLD2.JLDFile, Nothing}
    isopen::Bool
    scratch::Dict{Symbol, Any}  # Scratch for this simulation only
end

"""
    open(backend::ParallelJLD2Storage, sim_id::Integer; mode="a+")

Open a handle for the `sim_id`-th simulation's JLD2 file. The handle owns the scratch
namespace for that specific simulation.
"""
function Base.open(backend::ParallelJLD2Storage, sim_id::Integer; mode::AbstractString = "a+")
    sim_file_path = _sim_file_path(backend.path, sim_id)
    file = JLD2.jldopen(sim_file_path, mode)
    handle = ParallelJLD2StorageHandle(backend, Int(sim_id), file, true, Dict{Symbol, Any}())
    finalizer(close, handle)
    return handle
end

function Base.close(handle::ParallelJLD2StorageHandle)
    empty!(handle.scratch)  # Clear scratch for this simulation
    if handle.isopen && handle.file !== nothing
        JLD2.close(handle.file)
        handle.file = nothing
        handle.isopen = false
    end
    return nothing
end

# Public API for checking handle state
SimulationBasedInference.isopen(handle::ParallelJLD2StorageHandle) = handle.isopen

# --- Helper functions ---

"""
    _sim_file_path(path::String, sim_id::Integer)

Construct the path to the JLD2 file for simulation `sim_id`.
"""
function _sim_file_path(path::String, sim_id::Integer)
    # Use zero-padded filenames for consistent sorting in simulations/ subdirectory
    filename = @sprintf("simulation_%04d.jld2", sim_id)
    return joinpath(path, filename)
end

# --- shared file helpers (operate on an open JLD2 file) ---
_fkey(group, name) = "$group/$name"

function _ensure_scratch!(d::Dict{Symbol,Any}, name::Symbol)
    if !haskey(d, name)
        d[name] = Any[]
    end
    return d[name]
end

function _file_store!(file, group::Symbol, name::Symbol, x)
    key = _fkey(group, name)
    n = haskey(file, key) ? length(keys(file[key])) : 0
    file["$key/$(n + 1)"] = x
    return nothing
end

_file_get(file, group::Symbol, name::Symbol, j::Integer) = file["$(_fkey(group, name))/$j"]

function _file_length(file, group::Symbol, name::Symbol)
    k = _fkey(group, name)
    haskey(file, k) ? length(keys(file[k])) : 0
end

function _file_names(file, group::Symbol)
    g = "$group"
    return haskey(file, g) ? sort!(Symbol.(collect(keys(file[g])))) : Symbol[]
end

_file_has(file, group::Symbol, name::Symbol) = haskey(file, _fkey(group, name))

function _file_clear!(file, group::Symbol, name::Symbol)
    k = _fkey(i, group, name)
    haskey(file, k) && delete!(file, k)
    return nothing
end

function _file_clear_group!(file, group::Symbol)
    g = "$group"
    haskey(file, g) && delete!(file, g)
end

"""
    _read_metadata(file)

Read metadata dictionary from the current simulation's file.
"""
function _read_metadata(file)
    if haskey(file, "metadata")
        grp = file["metadata"]
        return Dict((Symbol(k) => file["metadata/$k"] for k in keys(grp)))
    else
        return Dict{Symbol, Any}()
    end
end

# --- handle-based operations (fast path - uses the open file for a single simulation) ---

SimulationBasedInference.num_simulations(h::ParallelJLD2StorageHandle) = h.backend.num_simulations

"""
    allocate!(handle::ParallelJLD2StorageHandle, input; metadata...)

Allocate a new simulation in the backend and return its ID. This should only be called
when the handle is opened for a fresh (non-existent) simulation file.
"""
function SimulationBasedInference.allocate!(h::ParallelJLD2StorageHandle, input; metadata...)
    # Store input
    h.file["input"] = input
    
    # Store metadata
    for kv in metadata
        h.file["metadata/$(first(kv))"] = last(kv)
    end
    
    return h.sim_id
end

SimulationBasedInference.getinputs(h::ParallelJLD2StorageHandle, ::Integer) = h.file["input"]

function SimulationBasedInference.setinputs!(h::ParallelJLD2StorageHandle, ::Integer, x)
    # JLD2 allows overwriting at close time
    h.file["input"] = x
    return h
end

SimulationBasedInference.getmetadata(h::ParallelJLD2StorageHandle, ::Integer) = _read_metadata(h.file)

function SimulationBasedInferenceJLD2Ext.setmetadata!(h::ParallelJLD2StorageHandle, ::Integer; kwargs...)
    old_metadata = _read_metadata(h.file)
    
    # Delete existing metadata keys
    if haskey(h.file, "metadata")
        for k in keys(old_metadata)
            delete!(h.file, "metadata/$k")
        end
    end
    
    # Write new/merged metadata
    for kv in merge(old_metadata, kwargs)
        h.file["metadata/$(first(kv))"] = last(kv)
    end
    
    return h
end

# output series (always on disk)
SimulationBasedInference.ensure_output!(::ParallelJLD2StorageHandle, ::Integer, ::Symbol) = nothing
SimulationBasedInference.store_output!(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol, x) = _file_store!(h.file, :outputs, name, x)
SimulationBasedInference.get_output(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol, j::Integer) = _file_get(h.file, :outputs, name, j)
SimulationBasedInference.get_outputs(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol) = [_file_get(h.file, :outputs, name, j) for j in 1:_file_length(h.file, :outputs, name)]
SimulationBasedInference.output_length(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol) = _file_length(h.file, :outputs, name)
SimulationBasedInference.output_names(h::ParallelJLD2StorageHandle, ::Integer) = _file_names(h.file, :outputs)
SimulationBasedInference.has_output(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol) = _file_has(h.file, :outputs, name)
SimulationBasedInference.empty_output!(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol) = _file_clear!(h.file, :outputs, name)

# scratch series (in-memory by default, on disk when scratch_in_memory == false)
# Scratch is per-simulation (owned by this handle)
function SimulationBasedInference.ensure_scratch!(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol)
    if h.backend.scratch_in_memory
        haskey(h.scratch, name) || (h.scratch[name] = Any[])
    end
    return nothing
end

function SimulationBasedInference.store_scratch!(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol, x)
    if h.backend.scratch_in_memory
        haskey(h.scratch, name) || (h.scratch[name] = Any[])
        push!(h.scratch[name], x)
    else
        _file_store!(h.file, :scratch, name, x)
    end
    return nothing
end

function SimulationBasedInference.get_scratch(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol, j::Integer)
    if h.backend.scratch_in_memory
        return h.scratch[name][j]
    else
        return _file_get(h.file, :scratch, name, j)
    end
end

function SimulationBasedInference.scratch_length(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol)
    if h.backend.scratch_in_memory
        return haskey(h.scratch, name) ? length(h.scratch[name]) : 0
    else
        return _file_length(h.file, :scratch, name)
    end
end

function SimulationBasedInference.scratch_names(h::ParallelJLD2StorageHandle, ::Integer)
    if h.backend.scratch_in_memory
        return collect(keys(h.scratch))
    else
        return _file_names(h.file, :scratch)
    end
end

function SimulationBasedInference.has_scratch(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol)
    if h.backend.scratch_in_memory
        return haskey(h.scratch, name)
    else
        return _file_has(h.file, :scratch, name)
    end
end

function SimulationBasedInference.empty_scratch!(h::ParallelJLD2StorageHandle, ::Integer, name::Symbol)
    if h.backend.scratch_in_memory
        haskey(h.scratch, name) && empty!(h.scratch[name])
    else
        _file_clear!(h.file, :scratch, name)
    end
    return nothing
end

"""
    empty!(handle::ParallelJLD2StorageHandle, ::Integer)

Clear all outputs and scratch for this simulation.
"""
function Base.empty!(h::ParallelJLD2StorageHandle, ::Integer)
    _file_clear_group!(h.file, :outputs)
    h.backend.scratch_in_memory || _file_clear_group!(h.file, :scratch)
    return nothing
end

# --- backend operations (slow path - auto-open/close per simulation) ---

SimulationBasedInference.num_simulations(b::ParallelJLD2Storage) = b.num_simulations

"""
    allocate!(backend::ParallelJLD2Storage, input; metadata...)

Allocate a new simulation in the backend. This creates a fresh JLD2 file for the simulation
and returns its ID.
"""
function SimulationBasedInference.allocate!(b::ParallelJLD2Storage, input; metadata...)
    # Increment counter and get new simulation ID
    b.num_simulations += 1
    sim_id = b.num_simulations
    
    # Create the simulation file
    handle = open(b, sim_id)
    allocate!(handle, input; metadata...)
    
    return sim_id
end

"""
    getinputs(backend::ParallelJLD2Storage, i::Integer)

Get the input for simulation `i`.
"""
SimulationBasedInference.getinputs(b::ParallelJLD2Storage, i::Integer) = open(b, i) do h; getinputs(h, i); end

"""
    setinputs!(backend::ParallelJLD2Storage, i::Integer, x)

Set the input for simulation `i`.
"""
SimulationBasedInference.setinputs!(b::ParallelJLD2Storage, i::Integer, x) = open(b, i) do h; setinputs!(h, i, x); end

"""
    getmetadata(backend::ParallelJLD2Storage, i::Integer)

Get the metadata for simulation `i`.
"""
SimulationBasedInference.getmetadata(b::ParallelJLD2Storage, i::Integer) = open(b, i) do h; getmetadata(h, i); end

"""
    setmetadata!(backend::ParallelJLD2Storage, i::Integer; kwargs...)

Set metadata for simulation `i`.
"""
SimulationBasedInference.setmetadata!(b::ParallelJLD2Storage, i::Integer; kwargs...) = open(b, i) do h; setmetadata!(h, i; kwargs...); end

"""
    ensure_output!(backend::ParallelJLD2Storage, i::Integer, name::Symbol)

Ensure output series `name` exists for simulation `i`.
"""
SimulationBasedInference.ensure_output!(b::ParallelJLD2Storage, i::Integer, name::Symbol) = open(b, i) do h; ensure_output!(h, i, name); end

"""
    store_output!(backend::ParallelJLD2Storage, i::Integer, name::Symbol, x)

Append `x` to output series `name` for simulation `i`.
"""
SimulationBasedInference.store_output!(b::ParallelJLD2Storage, i::Integer, name::Symbol, x) = open(b, i) do h; store_output!(h, i, name, x); end

"""
    get_output(backend::ParallelJLD2Storage, i::Integer, name::Symbol, j::Integer)

Get the `j`-th element of output series `name` for simulation `i`.
"""
SimulationBasedInference.get_output(b::ParallelJLD2Storage, i::Integer, name::Symbol, j::Integer) = open(b, i) do h; get_output(h, i, name, j); end

"""
    get_outputs(backend::ParallelJLD2Storage, i::Integer, name::Symbol)

Get all elements of output series `name` for simulation `i`.
"""
SimulationBasedInference.get_outputs(b::ParallelJLD2Storage, i::Integer, name::Symbol) = open(b, i) do h; get_outputs(h, i, name); end

"""
    output_length(backend::ParallelJLD2Storage, i::Integer, name::Symbol)

Get the length of output series `name` for simulation `i`.
"""
SimulationBasedInference.output_length(b::ParallelJLD2Storage, i::Integer, name::Symbol) = open(b, i) do h; output_length(h, i, name); end

"""
    output_names(backend::ParallelJLD2Storage, i::Integer)

Get all output series names for simulation `i`.
"""
SimulationBasedInference.output_names(b::ParallelJLD2Storage, i::Integer) = open(b, i) do h; output_names(h, i); end

"""
    has_output(backend::ParallelJLD2Storage, i::Integer, name::Symbol)

Check if output series `name` exists for simulation `i`.
"""
SimulationBasedInference.has_output(b::ParallelJLD2Storage, i::Integer, name::Symbol) = open(b, i) do h; has_output(h, i, name); end

"""
    empty_output!(backend::ParallelJLD2Storage, i::Integer, name::Symbol)

Clear output series `name` for simulation `i`.
"""
SimulationBasedInference.empty_output!(b::ParallelJLD2Storage, i::Integer, name::Symbol) = open(b, i) do h; empty_output!(h, i, name); end

"""
    empty!(backend::ParallelJLD2Storage, i::Integer)

Clear all outputs and scratch for simulation `i`.
"""
Base.empty!(b::ParallelJLD2Storage, i::Integer) = open(b, i) do h; empty!(h, i); end

"""
    empty!(backend::ParallelJLD2Storage)

Truncate the entire storage, removing all simulations.
"""
function Base.empty!(b::ParallelJLD2Storage)
    b.num_simulations = 0
    # Remove and recreate the output directory
    sim_dir = joinpath(b.path)
    if isdir(sim_dir)
        rm(sim_dir; recursive=true)
    end
    mkpath(sim_dir)
    return nothing
end

end
