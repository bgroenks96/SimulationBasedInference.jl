module SimulationBasedInferenceJLD2Ext

using SimulationBasedInference
using SimulationBasedInference: StorageBackend, InMemoryStorage, SimulationDataSet,
    num_simulations, allocate!, getinputs, setinputs!, getmetadata,
    store_output!, get_output, output_length, output_names, has_output, ensure_output!, empty_output!,
    store_scratch!, get_scratch_buffer, scratch_length, scratch_names, has_scratch, ensure_scratch!, empty_scratch!

using FileIO
using JLD2

"""
    JLD2Storage <: StorageBackend

Disk-backed storage backend that persists each simulation as a group in a JLD2 file.
Simulation outputs are written directly to disk as they are accumulated.

Scratch (transient working) storage can be either kept in memory when `scratch_in_memory`
is set to `true`, or otherwise written to disk.

Layout per simulation `i`:

    simulations/<i>/input
    simulations/<i>/metadata
    simulations/<i>/outputs/<name>/<j>     # j = 1-based element index
    simulations/<i>/scratch/<name>/<j>     # only when scratch_in_memory == false

`num_simulations` is the number of subgroups of `simulations`.
"""
mutable struct JLD2Storage{scratchMemory} <: StorageBackend
    path::String
    num_simulations::Int
    scratch::scratchMemory
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
    if scratch_in_memory
        scratch = InMemoryStorage()
        # keep the in-memory scratch store index-aligned with the on-disk simulations
        for _ in 1:n
            allocate!(scratch, nothing)
        end
        return JLD2Storage(String(path), n, scratch)
    else
        return JLD2Storage(String(path), n, nothing)
    end
end

"""
    OnDiskSimulationDataSet(file::File{format"JLD2"}; overwrite=false, scratch_in_memory=true)

Construct a `SimulationDataSet` backed by a [`JLD2Storage`](@ref) at `file`.
"""
SimulationBasedInference.OnDiskSimulationDataSet(file::File{format"JLD2"}; kwargs...) =
    SimulationDataSet(JLD2Storage(file.filename; kwargs...))

scratch_in_memory(storage::JLD2Storage) = !isnothing(storage.scratch)

_key(i, group, name) = "simulations/$i/$group/$name"

function _disk_store!(backend::JLD2Storage, i::Integer, group::Symbol, name::Symbol, x)
    JLD2.jldopen(backend.path, "a+") do file
        key = _key(i, group, name)
        n = haskey(file, key) ? length(keys(file[key])) : 0
        file["$key/$(n + 1)"] = x
    end
    return backend
end

function _disk_get(backend::JLD2Storage, i::Integer, group::Symbol, name::Symbol, j::Integer)
    JLD2.jldopen(backend.path, "r") do file
        file["$(_key(i, group, name))/$j"]
    end
end

function _disk_length(backend::JLD2Storage, i::Integer, group::Symbol, name::Symbol)
    JLD2.jldopen(backend.path, "r") do file
        key = _key(i, group, name)
        haskey(file, key) ? length(keys(file[key])) : 0
    end
end

function _disk_names(backend::JLD2Storage, i::Integer, group::Symbol)
    JLD2.jldopen(backend.path, "r") do file
        g = "simulations/$i/$group"
        haskey(file, g) ? sort!(Symbol.(collect(keys(file[g])))) : Symbol[]
    end
end

function _disk_has(backend::JLD2Storage, i::Integer, group::Symbol, name::Symbol)
    JLD2.jldopen(backend.path, "r") do file
        haskey(file, _key(i, group, name))
    end
end

function _disk_clear!(backend::JLD2Storage, i::Integer, group::Symbol, name::Symbol)
    JLD2.jldopen(backend.path, "a+") do file
        key = _key(i, group, name)
        haskey(file, key) && delete!(file, key)
    end
    return backend
end

# --- StorageBackend interface ---

SimulationBasedInference.num_simulations(backend::JLD2Storage) = backend.num_simulations

function SimulationBasedInference.allocate!(backend::JLD2Storage, input; metadata...)
    i = backend.num_simulations + 1
    JLD2.jldopen(backend.path, "a+") do file
        prefix = "simulations/$i"
        file["$prefix/input"] = input
        file["$prefix/metadata"] = Dict{Symbol,Any}(metadata)
    end
    allocate!(backend.scratch, nothing)  # parallel in-memory scratch slot (index i)
    backend.num_simulations = i
    return i
end

SimulationBasedInference.getinputs(backend::JLD2Storage, i::Integer) =
    JLD2.jldopen(backend.path, "r") do file
        file["simulations/$i/input"]
    end

function SimulationBasedInference.setinputs!(backend::JLD2Storage, i::Integer, x)
    JLD2.jldopen(backend.path, "a+") do file
        key = "simulations/$i/input"
        haskey(file, key) && delete!(file, key)  # JLD2 datasets are write-once; overwrite by delete+write
        file[key] = x
    end
    return backend
end

SimulationBasedInference.getmetadata(backend::JLD2Storage, i::Integer) =
    JLD2.jldopen(backend.path, "r") do file
        file["simulations/$i/metadata"]
    end

# --- output storage (always on disk) ---
SimulationBasedInference.ensure_output!(backend::JLD2Storage, i::Integer, name::Symbol) = nothing
SimulationBasedInference.store_output!(backend::JLD2Storage, i::Integer, name::Symbol, x) = _disk_store!(backend, i, :outputs, name, x)
SimulationBasedInference.get_output(backend::JLD2Storage, i::Integer, name::Symbol, j::Integer) = _disk_get(backend, i, :outputs, name, j)
SimulationBasedInference.output_length(backend::JLD2Storage, i::Integer, name::Symbol) = _disk_length(backend, i, :outputs, name)
SimulationBasedInference.output_names(backend::JLD2Storage, i::Integer) = _disk_names(backend, i, :outputs)
SimulationBasedInference.has_output(backend::JLD2Storage, i::Integer, name::Symbol) = _disk_has(backend, i, :outputs, name)
SimulationBasedInference.empty_output!(backend::JLD2Storage, i::Integer, name::Symbol) = _disk_clear!(backend, i, :outputs, name)

# --- scratch storage (in memory by default, optionally on disk) ---
SimulationBasedInference.ensure_scratch!(backend::JLD2Storage, i::Integer, name::Symbol) =
    scratch_in_memory(backend) ? ensure_scratch!(backend.scratch, i, name) : nothing
SimulationBasedInference.store_scratch!(backend::JLD2Storage, i::Integer, name::Symbol, x) =
    scratch_in_memory(backend) ? store_scratch!(backend.scratch, i, name, x) : _disk_store!(backend, i, :scratch, name, x)
SimulationBasedInference.scratch_length(backend::JLD2Storage, i::Integer, name::Symbol) =
    scratch_in_memory(backend) ? scratch_length(backend.scratch, i, name) : _disk_length(backend, i, :scratch, name)
SimulationBasedInference.scratch_names(backend::JLD2Storage, i::Integer) =
    scratch_in_memory(backend) ? scratch_names(backend.scratch, i) : _disk_names(backend, i, :scratch)
SimulationBasedInference.has_scratch(backend::JLD2Storage, i::Integer, name::Symbol) =
    scratch_in_memory(backend) ? has_scratch(backend.scratch, i, name) : _disk_has(backend, i, :scratch, name)
SimulationBasedInference.empty_scratch!(backend::JLD2Storage, i::Integer, name::Symbol) =
    (scratch_in_memory(backend) ? empty_scratch!(backend.scratch, i, name) : _disk_clear!(backend, i, :scratch, name); backend)

function Base.empty!(backend::JLD2Storage, i::Integer)
    JLD2.jldopen(backend.path, "a+") do file
        for group in ("outputs", "scratch")
            g = "simulations/$i/$group"
            haskey(file, g) && delete!(file, g)
        end
    end
    scratch_in_memory(backend) && empty!(backend.scratch, i)
    return backend
end

function Base.empty!(backend::JLD2Storage)
    backend.num_simulations = 0
    JLD2.jldopen(backend.path, "w") do _ end  # truncate
    scratch_in_memory(backend) && empty!(backend.scratch)
    return backend
end

end
