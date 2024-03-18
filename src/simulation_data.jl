"""
    SimulationData

Base type representing storage for simulation data, i.e. input/output pairs.
"""
abstract type SimulationData end

"""
    store!(::SimulationData, x::AbstractVector, y)

Stores the input/output pair x/y in the given forward map storage container.
"""
store!(::SimulationData, x::AbstractVector, y) = error("not implemented")
function store!(storage::SimulationData, X::AbstractMatrix, y::AbstractVector; attr...)
    for (i, x) in enumerate(eachcol(X))
        store!(storage, x, y[i]; attr...)
    end
end

Base.length(storage::SimulationData) = length(storage.inputs)
Base.lastindex(storage::SimulationData) = length(storage)
Base.firstindex(storage::SimulationData) = 1

"""
    SimulationArrayStorage <: SimulationData

Simple implementation of `SimulationData` that stores all results in generically
typed `Vector`s.
"""
struct SimulationArrayStorage <: SimulationData
    inputs::Vector
    outputs::Vector
    metadata::Vector
end

SimulationArrayStorage() = SimulationArrayStorage([], [], [])

Base.getindex(storage::SimulationArrayStorage, i) = (storage.inputs[i], storage.outputs[i], storage.metadata[i])

Base.iterate(storage::SimulationArrayStorage) = (storage[1], 1)
Base.iterate(storage::SimulationArrayStorage, state) = state < length(storage) ? (storage[state], state+1) : (storage[end], nothing)

getinputs(storage::SimulationArrayStorage) = storage.inputs
getinputs(storage::SimulationArrayStorage, i) = storage.inputs[i]

getoutputs(storage::SimulationArrayStorage) = storage.outputs
getoutputs(storage::SimulationArrayStorage, i) = storage.outputs[i]

getmetadata(storage::SimulationArrayStorage) = storage.metadata
getmetadata(storage::SimulationArrayStorage, i) = storage.metadata[i]

function store!(storage::SimulationArrayStorage, x::AbstractVector, y; attr...)
    push!(storage.inputs, x)
    push!(storage.outputs, y)
    push!(storage.metadata, (; attr...))
end

function clear!(storage::SimulationArrayStorage)
    resize!(storage.inputs, 0)
    resize!(storage.outputs, 0)
    resize!(storage.metadata, 0)
end
