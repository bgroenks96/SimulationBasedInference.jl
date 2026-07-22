"""
    SimulationData{B}

A view of the data for a single simulation stored in the given [`StorageBackend`](@ref).
The optional `handle` field maintains a persistent storage handle when needed (e.g., for
observables that require scratch storage across multiple `observe!` calls).
"""
struct SimulationData{B<:StorageBackend}
    backend::B
    index::Int
    handle::Ref{Union{Nothing, StorageHandle}}  # persistent handle for buffer operations
end

# Constructor without explicit handle (lazy initialization)
SimulationData(backend::StorageBackend, index::Int) = SimulationData(backend, index, Ref{Union{Nothing, StorageHandle}}(nothing))

"""
Allocate and construct a new `SimulationData` from the given `backend`, or create a new [`InMemoryStorage`](@ref)
backend if not specified.
"""
function SimulationData(backend::StorageBackend=InMemoryStorage(); inputs=nothing, metadata...)
    i = allocate!(backend, inputs; metadata...)
    return SimulationData(backend, i)
end

getinputs(data::SimulationData) = getinputs(data.backend, data.index)
setinputs!(data::SimulationData, x) = setinputs!(data.backend, data.index, x)
getmetadata(data::SimulationData) = getmetadata(data.backend, data.index)
setmetadata!(data::SimulationData; kwargs...) = setmetadata!(data.backend, data.index; kwargs...)

output_names(data::SimulationData) = output_names(data.backend, data.index)

has_output(data::SimulationData, name::Symbol) = has_output(data.backend, data.index, name)

"""
    store!(data::SimulationData, name::Symbol, value)

Append `value` to the persistent output series for observable `name`.
"""
store!(data::SimulationData, name::Symbol, value) = store_output!(data.backend, data.index, name, value)

"""
    getoutput(data::SimulationData, name::Symbol)

Return the collected output sequence (a `Vector`) for observable `name`.
"""
getoutput(data::SimulationData, name::Symbol) = get_outputs(data.backend, data.index, name)

"""
    getoutputs(data::SimulationData)

Return a `NamedTuple` mapping each output name to its corresponding data series.
"""
getoutputs(data::SimulationData) = (; (nm => getoutput(data, nm) for nm in output_names(data))...)

"""
    get_output_buffer(data::SimulationData, name::Symbol)

Return a [`DataBuffer`](@ref) view of the output data for variable `name`, creating it if necessary.
The handle is opened lazily and reused across calls until explicitly closed.
"""
function get_output_buffer(data::SimulationData, name::Symbol)
    # Open persistent handle if not already open
    if isnothing(data.handle[]) || !isopen(data.handle[])
        data.handle[] = open(data.backend, data.index)
    end
    ensure_output!(data.handle[], name)
    return DataBuffer{:output}(data.handle[], name)
end

"""
    with_output_buffer(func!, data::SimulationData, name::Symbol)

Open an output [`DataBuffer`](@ref) for `name`, call `func!(buffer)`, then close
the handle. Equivalent to calling [`get_output_buffer`](@ref), using the result,
and then calling `close(data)`.
"""
function with_output_buffer(func!, data::SimulationData, name::Symbol)
    buffer = get_output_buffer(data, name)
    try
        return func!(buffer)
    finally
        close(data)
    end
end

"""
    get_scratch_buffer(data::SimulationData, name::Symbol=:scratch)

Return a [`DataBuffer`](@ref) view of the transient scratch series `name`.
The handle is opened lazily and reused across calls until explicitly closed.
"""
function get_scratch_buffer(data::SimulationData, name::Symbol=:scratch)
    # Open persistent handle if not already open
    if isnothing(data.handle[]) || !isopen(data.handle[])
        data.handle[] = open(data.backend, data.index)
    end
    ensure_scratch!(data.handle[], name)
    return DataBuffer{:scratch}(data.handle[], name)
end

"""
    with_scratch_buffer(func!, data::SimulationData, name::Symbol=:scratch)

Open a scratch [`DataBuffer`](@ref) for `name`, call `func!(buffer)`, then close
the handle. Equivalent to calling [`get_scratch_buffer`](@ref), using the result,
and then calling `close(data)`.
"""
function with_scratch_buffer(func!, data::SimulationData, name::Symbol=:scratch)
    buffer = get_scratch_buffer(data, name)
    try
        return func!(buffer)
    finally
        close(data)
    end
end

Base.empty!(data::SimulationData) = empty!(data.backend, data.index)

function Base.copy!(dest::SimulationData, src::SimulationData)
    if dest.backend != src.backend
        copy!(dest.backend, src.backend, src.index)
    end
end

"""
    close(data::SimulationData)

Close the persistent handle for this simulation if it is open. Should be called when the
simulation is complete to release file descriptors and flush any pending writes.
"""
function Base.close(data::SimulationData)
    if !isnothing(data.handle[]) && isopen(data.handle[])
        close(data.handle[])
        data.handle[] = nothing
    end
    return nothing
end

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

backend(storage::SimulationDataSet) = storage.backend

Base.length(storage::SimulationDataSet) = num_simulations(storage.backend)
Base.firstindex(::SimulationDataSet) = 1
Base.lastindex(storage::SimulationDataSet) = length(storage)
Base.isempty(storage::SimulationDataSet) = length(storage) == 0
Base.getindex(storage::SimulationDataSet, i::Integer) = SimulationData(storage.backend, i)

"""
    allocate!(storage::SimulationDataSet, inputs=nothing; metadata...)

Allocate simulation data storage in the underlying `backend` (with the given `input`) and return a
[`SimulationData`](@ref) view of it.
"""
function allocate!(storage::SimulationDataSet, inputs=nothing; metadata...)
    i = allocate!(storage.backend, inputs; metadata...)
    return SimulationData(storage.backend, i)
end

"""
    allocate!(storage::SimulationDataSet, inputs::AbstractMatrix; metadata...)

Allocate a new `SimulationData` on the storage backend for each column of `inputs`. Returns a vector
of [`SimulationData`](@ref) of length `size(inputs, 2)`.
"""
function allocate!(storage::SimulationDataSet, inputs::AbstractMatrix; metadata...)
    for x in eachcol(inputs)
        allocate!(storage, x; metadata...)
    end
    return [SimulationData(storage.backend, i) for i in 1:size(inputs, 2)]
end

"""
    store!(storage::SimulationDataSet, data::SimulationData; metadata...)

Append an existing `SimulationData` (possibly held in a different backend, e.g. a per-member
forward solve) to `storage`, copying its input and output series into the backing store and
merging any extra `metadata`.
"""
function store!(storage::SimulationDataSet, data::SimulationData; metadata...)
    # First allocate on the backend to get the simulation ID
    i = allocate!(storage.backend, getinputs(data); getmetadata(data)..., metadata...)
    
    # Then open a handle for that specific simulation and copy outputs
    open(storage.backend, i) do handle
        for nm in output_names(data)
            # Get all values from source and store them
            for value in getoutput(data, nm)
                store_output!(handle, nm, value)
            end
        end
    end
    
    return SimulationData(storage.backend, i)
end

Base.empty!(storage::SimulationDataSet) = empty!(storage.backend)

# iterating a storage yields (input, outputs, metadata) triples, one per simulation
function Base.iterate(storage::SimulationDataSet, i::Int=1)
    return i <= length(storage) ? ((getinputs(storage, i), getoutputs(storage, i), getmetadata(storage, i)), i + 1) : nothing
end

getinputs(storage::SimulationDataSet, i::Integer) = getinputs(storage[i])
getoutputs(storage::SimulationDataSet, i::Integer) = getoutputs(storage[i])
getmetadata(storage::SimulationDataSet, i::Integer) = getmetadata(storage[i])

getinputs(storage::SimulationDataSet) = [getinputs(storage, i) for i in 1:length(storage)]
getoutputs(storage::SimulationDataSet) = [getoutputs(storage, i) for i in 1:length(storage)]
getmetadata(storage::SimulationDataSet) = [getmetadata(storage, i) for i in 1:length(storage)]

"""
    iterations(storage::SimulationDataSet)

Return the number of distinct `iter` values recorded in the simulations' metadata, i.e. the
number of inference iterations. Returns `0` for an empty storage and `1` when no `iter`
metadata is present.
"""
function iterations(storage::SimulationDataSet)
    length(storage) == 0 && return 0
    return maximum(get(getmetadata(storage, i), :iter, 1) for i in 1:length(storage))
end
