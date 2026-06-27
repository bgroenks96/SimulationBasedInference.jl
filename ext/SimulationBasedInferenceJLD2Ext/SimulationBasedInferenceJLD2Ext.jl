module SimulationBasedInferenceJLD2Ext

using SimulationBasedInference
using SimulationBasedInference: StorageBackend, InMemoryStorage, SimulationDataSet,
    num_simulations, allocate!, getinputs, setinputs!, getmetadata,
    store_output!, get_output, output_length, output_names, has_output, ensure_output!, empty_output!,
    store_scratch!, get_scratch_storage, scratch_length, scratch_names, has_scratch, ensure_scratch!, empty_scratch!

using FileIO
using JLD2

"""
    JLD2Storage <: StorageBackend

Disk-backed storage backend that persists each simulation as a group in a JLD2 file.
Simulations are written directly to disk as they are accumulated, without any in-memory
buffering of completed simulations.

Scratch (transient working) series have a `scratch_in_memory` switch: when `true` (default)
they are held in the embedded `scratch::InMemoryStorage` and never written to disk; when
`false` they are written to disk alongside the outputs. In either case the `scratch` field is
present and kept index-aligned with the on-disk simulations.

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

# --- shared on-disk series helpers (1-based element keys; names derived from group keys) ---
_series(i, group, name) = "simulations/$i/$group/$name"

function _disk_store!(b::JLD2Storage, i::Integer, group::Symbol, name::Symbol, x)
    JLD2.jldopen(b.path, "a+") do file
        series = _series(i, group, name)
        n = haskey(file, series) ? length(keys(file[series])) : 0
        file["$series/$(n + 1)"] = x
    end
    return b
end

function _disk_get(b::JLD2Storage, i::Integer, group::Symbol, name::Symbol, j::Integer)
    JLD2.jldopen(b.path, "r") do file
        file["$(_series(i, group, name))/$j"]
    end
end

function _disk_length(b::JLD2Storage, i::Integer, group::Symbol, name::Symbol)
    JLD2.jldopen(b.path, "r") do file
        series = _series(i, group, name)
        haskey(file, series) ? length(keys(file[series])) : 0
    end
end

function _disk_names(b::JLD2Storage, i::Integer, group::Symbol)
    JLD2.jldopen(b.path, "r") do file
        g = "simulations/$i/$group"
        haskey(file, g) ? sort!(Symbol.(collect(keys(file[g])))) : Symbol[]
    end
end

function _disk_has(b::JLD2Storage, i::Integer, group::Symbol, name::Symbol)
    JLD2.jldopen(b.path, "r") do file
        haskey(file, _series(i, group, name))
    end
end

function _disk_clear!(b::JLD2Storage, i::Integer, group::Symbol, name::Symbol)
    JLD2.jldopen(b.path, "a+") do file
        series = _series(i, group, name)
        haskey(file, series) && delete!(file, series)
    end
    return b
end

# --- StorageBackend interface ---

SimulationBasedInference.num_simulations(b::JLD2Storage) = b.num_simulations

function SimulationBasedInference.allocate!(b::JLD2Storage, input; metadata...)
    i = b.num_simulations + 1
    JLD2.jldopen(b.path, "a+") do file
        prefix = "simulations/$i"
        file["$prefix/input"] = input
        file["$prefix/metadata"] = Dict{Symbol,Any}(metadata)
    end
    allocate!(b.scratch, nothing)  # parallel in-memory scratch slot (index i)
    b.num_simulations = i
    return i
end

SimulationBasedInference.getinputs(b::JLD2Storage, i::Integer) =
    JLD2.jldopen(b.path, "r") do file
        file["simulations/$i/input"]
    end

function SimulationBasedInference.setinputs!(b::JLD2Storage, i::Integer, x)
    JLD2.jldopen(b.path, "a+") do file
        key = "simulations/$i/input"
        haskey(file, key) && delete!(file, key)  # JLD2 datasets are write-once; overwrite by delete+write
        file[key] = x
    end
    return b
end

SimulationBasedInference.getmetadata(b::JLD2Storage, i::Integer) =
    JLD2.jldopen(b.path, "r") do file
        file["simulations/$i/metadata"]
    end

# --- output series (always on disk) ---
SimulationBasedInference.ensure_output!(b::JLD2Storage, i::Integer, name::Symbol) = nothing
SimulationBasedInference.store_output!(b::JLD2Storage, i::Integer, name::Symbol, x) = _disk_store!(b, i, :outputs, name, x)
SimulationBasedInference.get_output(b::JLD2Storage, i::Integer, name::Symbol, j::Integer) = _disk_get(b, i, :outputs, name, j)
SimulationBasedInference.output_length(b::JLD2Storage, i::Integer, name::Symbol) = _disk_length(b, i, :outputs, name)
SimulationBasedInference.output_names(b::JLD2Storage, i::Integer) = _disk_names(b, i, :outputs)
SimulationBasedInference.has_output(b::JLD2Storage, i::Integer, name::Symbol) = _disk_has(b, i, :outputs, name)
SimulationBasedInference.empty_output!(b::JLD2Storage, i::Integer, name::Symbol) = _disk_clear!(b, i, :outputs, name)

# --- scratch series (in memory by default, optionally on disk) ---
SimulationBasedInference.ensure_scratch!(b::JLD2Storage, i::Integer, name::Symbol) =
    scratch_in_memory(b) ? ensure_scratch!(b.scratch, i, name) : nothing
SimulationBasedInference.store_scratch!(b::JLD2Storage, i::Integer, name::Symbol, x) =
    scratch_in_memory(b) ? store_scratch!(b.scratch, i, name, x) : _disk_store!(b, i, :scratch, name, x)
SimulationBasedInference.get_scratch_storage(b::JLD2Storage, i::Integer, name::Symbol, j::Integer) =
    scratch_in_memory(b) ? get_scratch_storage(b.scratch, i, name, j) : _disk_get(b, i, :scratch, name, j)
SimulationBasedInference.scratch_length(b::JLD2Storage, i::Integer, name::Symbol) =
    scratch_in_memory(b) ? scratch_length(b.scratch, i, name) : _disk_length(b, i, :scratch, name)
SimulationBasedInference.scratch_names(b::JLD2Storage, i::Integer) =
    scratch_in_memory(b) ? scratch_names(b.scratch, i) : _disk_names(b, i, :scratch)
SimulationBasedInference.has_scratch(b::JLD2Storage, i::Integer, name::Symbol) =
    scratch_in_memory(b) ? has_scratch(b.scratch, i, name) : _disk_has(b, i, :scratch, name)
SimulationBasedInference.empty_scratch!(b::JLD2Storage, i::Integer, name::Symbol) =
    (scratch_in_memory(b) ? empty_scratch!(b.scratch, i, name) : _disk_clear!(b, i, :scratch, name); b)

# --- whole-simulation / whole-backend ---
function Base.empty!(b::JLD2Storage, i::Integer)
    JLD2.jldopen(b.path, "a+") do file
        for group in ("outputs", "scratch")
            g = "simulations/$i/$group"
            haskey(file, g) && delete!(file, g)
        end
    end
    scratch_in_memory(b) && empty!(b.scratch, i)
    return b
end

function Base.empty!(b::JLD2Storage)
    b.num_simulations = 0
    JLD2.jldopen(b.path, "w") do _ end  # truncate
    scratch_in_memory(b) && empty!(b.scratch)
    return b
end

end
