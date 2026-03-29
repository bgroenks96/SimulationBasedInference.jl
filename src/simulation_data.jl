"""
    SimulationData{inputType, outputType}

Base type representing storage for simulation data, i.e. input/output pairs.
"""
abstract type SimulationData{inputType, outputType} end

"""
    store!(::SimulationData, x, y)

Stores the input/output pair x/y in the given forward map storage container.
"""
function store! end

Base.length(storage::SimulationData) = length(storage.inputs)
Base.lastindex(storage::SimulationData) = length(storage)
Base.firstindex(storage::SimulationData) = 1

"""
    SimulationArrayStorage <: SimulationData

Simple implementation of `SimulationData` that stores all results in generically
typed `Vector`s.
"""
struct SimulationArrayStorage{inputType, outputType, metaType} <: SimulationData{inputType, outputType}
    inputs::Vector{inputType}
    outputs::Vector{outputType}
    metadata::Vector{metaType}
end

SimulationArrayStorage(;
    input_type::Type = Any,
    output_type::Type = Any,
    metadata_type::Type = Any
) = SimulationArrayStorage(input_type[], output_type[], metadata_type[])

Base.getindex(storage::SimulationArrayStorage, i) = (storage.inputs[i], storage.outputs[i], storage.metadata[i])

Base.iterate(storage::SimulationArrayStorage) = (storage[1], 1)
Base.iterate(storage::SimulationArrayStorage, state) = state <= length(storage) ? (storage[state], state+1) : nothing

getinputs(storage::SimulationArrayStorage) = storage.inputs
getinputs(storage::SimulationArrayStorage, i) = storage.inputs[i]

getoutputs(storage::SimulationArrayStorage) = storage.outputs
getoutputs(storage::SimulationArrayStorage, i) = storage.outputs[i]

getmetadata(storage::SimulationArrayStorage) = storage.metadata
getmetadata(storage::SimulationArrayStorage, i) = storage.metadata[i]

function store!(storage::SimulationArrayStorage, x, y; attr...)
    push!(storage.inputs, x)
    push!(storage.outputs, y)
    push!(storage.metadata, (; attr...))
end

function clear!(storage::SimulationArrayStorage)
    resize!(storage.inputs, 0)
    resize!(storage.outputs, 0)
    resize!(storage.metadata, 0)
end
