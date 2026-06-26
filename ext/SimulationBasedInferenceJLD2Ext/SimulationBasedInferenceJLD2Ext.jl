module SimulationBasedInferenceJLD2Ext

using SimulationBasedInference
using SimulationBasedInference: StorageBackend, InMemoryStorage, SimulationDataSet,
    num_simulations, allocate!, getinputs, setinputs!, getmetadata,
    store_output!, get_output, output_length, output_names, has_output, ensure_output!, clear_output!,
    store_scratch!, get_scratch, scratch_length, scratch_names, has_scratch, ensure_scratch!, clear_scratch!,
    clear_simulation!

using JLD2

"""
    JLD2Storage <: StorageBackend

Disk-backed storage backend that persists each simulation as a group in a JLD2 file.
Completed simulations are streamed to disk so that the full collection need not reside in
memory: only the simulation currently being accumulated is kept in RAM (in an embedded
`InMemoryStorage`), and already-persisted simulations are read back lazily (cached). Scratch
series are never persisted.

Layout per simulation `i`:

    simulations/<i>/input
    simulations/<i>/metadata
    simulations/<i>/output_names
    simulations/<i>/outputs/<name>

The persisted count is the number of subgroups of `simulations`.
"""
mutable struct JLD2Storage <: StorageBackend
    path::String
    persisted::Int
    pending::InMemoryStorage
    cache::Dict{Int,Any}
end

function _persisted_count(path::AbstractString)
    isfile(path) || return 0
    return JLD2.jldopen(path, "r") do file
        haskey(file, "simulations") ? length(keys(file["simulations"])) : 0
    end
end

function JLD2Storage(path::AbstractString; overwrite::Bool=false)
    if !isfile(path) || overwrite
        JLD2.jldopen(path, "w") do _ end  # create/truncate
        return JLD2Storage(String(path), 0, InMemoryStorage(), Dict{Int,Any}())
    end
    return JLD2Storage(String(path), _persisted_count(path), InMemoryStorage(), Dict{Int,Any}())
end

"""
    JLD2SimulationDataSet(path; overwrite=false)

Construct a `SimulationDataSet` backed by a [`JLD2Storage`](@ref) at `path`.
"""
SimulationBasedInference.JLD2SimulationDataSet(path::AbstractString; kwargs...) =
    SimulationDataSet(JLD2Storage(path; kwargs...))

# --- pending / persisted index helpers ---
_ispending(b::JLD2Storage, i::Integer) = i > b.persisted
_local(b::JLD2Storage, i::Integer) = i - b.persisted

function _load(b::JLD2Storage, i::Integer)
    haskey(b.cache, i) && return b.cache[i]
    loaded = JLD2.jldopen(b.path, "r") do file
        prefix = "simulations/$i"
        names = Symbol.(file["$prefix/output_names"])
        outputs = Dict{Symbol,Vector}()
        for nm in names
            outputs[nm] = collect(file["$prefix/outputs/$nm"])
        end
        (; input=file["$prefix/input"], metadata=file["$prefix/metadata"], names, outputs)
    end
    b.cache[i] = loaded
    return loaded
end

function _flush_pending!(b::JLD2Storage)
    np = num_simulations(b.pending)
    np == 0 && return b
    JLD2.jldopen(b.path, "a+") do file
        for j in 1:np
            i = b.persisted + j
            prefix = "simulations/$i"
            file["$prefix/input"] = getinputs(b.pending, j)
            file["$prefix/metadata"] = getmetadata(b.pending, j)
            names = output_names(b.pending, j)
            file["$prefix/output_names"] = names
            for nm in names
                n = output_length(b.pending, j, nm)
                file["$prefix/outputs/$nm"] = [get_output(b.pending, j, nm, k) for k in 1:n]
            end
        end
    end
    b.persisted += np
    SimulationBasedInference.clear!(b.pending)
    return b
end

# --- StorageBackend interface ---

SimulationBasedInference.num_simulations(b::JLD2Storage) = b.persisted + num_simulations(b.pending)

function SimulationBasedInference.allocate!(b::JLD2Storage, input; metadata...)
    _flush_pending!(b)                          # persist the previously-completed simulation(s)
    allocate!(b.pending, input; metadata...)    # start a fresh in-memory slot
    return b.persisted + num_simulations(b.pending)
end

SimulationBasedInference.getinputs(b::JLD2Storage, i::Integer) =
    _ispending(b, i) ? getinputs(b.pending, _local(b, i)) : _load(b, i).input

function SimulationBasedInference.setinputs!(b::JLD2Storage, i::Integer, x)
    _ispending(b, i) || error("cannot modify simulation $i: already persisted to disk")
    return setinputs!(b.pending, _local(b, i), x)
end

SimulationBasedInference.getmetadata(b::JLD2Storage, i::Integer) =
    _ispending(b, i) ? getmetadata(b.pending, _local(b, i)) : _load(b, i).metadata

# --- output series ---
function SimulationBasedInference.ensure_output!(b::JLD2Storage, i::Integer, name::Symbol)
    _ispending(b, i) && return ensure_output!(b.pending, _local(b, i), name)
    return nothing
end
function SimulationBasedInference.store_output!(b::JLD2Storage, i::Integer, name::Symbol, x)
    _ispending(b, i) || error("cannot write to simulation $i: already persisted to disk")
    return store_output!(b.pending, _local(b, i), name, x)
end
function SimulationBasedInference.get_output(b::JLD2Storage, i::Integer, name::Symbol, j::Integer)
    _ispending(b, i) && return get_output(b.pending, _local(b, i), name, j)
    return _load(b, i).outputs[name][j]
end
function SimulationBasedInference.output_length(b::JLD2Storage, i::Integer, name::Symbol)
    _ispending(b, i) && return output_length(b.pending, _local(b, i), name)
    loaded = _load(b, i)
    return haskey(loaded.outputs, name) ? length(loaded.outputs[name]) : 0
end
SimulationBasedInference.output_names(b::JLD2Storage, i::Integer) =
    _ispending(b, i) ? output_names(b.pending, _local(b, i)) : _load(b, i).names
SimulationBasedInference.has_output(b::JLD2Storage, i::Integer, name::Symbol) =
    _ispending(b, i) ? has_output(b.pending, _local(b, i), name) : name in _load(b, i).names
function SimulationBasedInference.clear_output!(b::JLD2Storage, i::Integer, name::Symbol)
    _ispending(b, i) && clear_output!(b.pending, _local(b, i), name)
    return b
end

# --- scratch series (never persisted; only available for the pending simulation) ---
function SimulationBasedInference.ensure_scratch!(b::JLD2Storage, i::Integer, name::Symbol)
    _ispending(b, i) && return ensure_scratch!(b.pending, _local(b, i), name)
    return nothing
end
function SimulationBasedInference.store_scratch!(b::JLD2Storage, i::Integer, name::Symbol, x)
    _ispending(b, i) || error("cannot write scratch to simulation $i: already persisted to disk")
    return store_scratch!(b.pending, _local(b, i), name, x)
end
SimulationBasedInference.get_scratch(b::JLD2Storage, i::Integer, name::Symbol, j::Integer) =
    get_scratch(b.pending, _local(b, i), name, j)
SimulationBasedInference.scratch_length(b::JLD2Storage, i::Integer, name::Symbol) =
    _ispending(b, i) ? scratch_length(b.pending, _local(b, i), name) : 0
SimulationBasedInference.scratch_names(b::JLD2Storage, i::Integer) =
    _ispending(b, i) ? scratch_names(b.pending, _local(b, i)) : Symbol[]
SimulationBasedInference.has_scratch(b::JLD2Storage, i::Integer, name::Symbol) =
    _ispending(b, i) ? has_scratch(b.pending, _local(b, i), name) : false
function SimulationBasedInference.clear_scratch!(b::JLD2Storage, i::Integer, name::Symbol)
    _ispending(b, i) && clear_scratch!(b.pending, _local(b, i), name)
    return b
end

# --- whole-simulation / whole-backend ---
function SimulationBasedInference.clear_simulation!(b::JLD2Storage, i::Integer)
    _ispending(b, i) && clear_simulation!(b.pending, _local(b, i))
    return b
end

function SimulationBasedInference.clear!(b::JLD2Storage)
    b.persisted = 0
    SimulationBasedInference.clear!(b.pending)
    empty!(b.cache)
    JLD2.jldopen(b.path, "w") do _ end  # truncate
    return b
end

SimulationBasedInference.flush!(b::JLD2Storage) = _flush_pending!(b)
Base.close(b::JLD2Storage) = (_flush_pending!(b); nothing)

end
