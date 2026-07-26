# [Simulation data storage](@id simulation_data)

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference
end
```

A key problem in simulation-based inference is handling output from the simulator. In the most trivial case of
forward map simulations, the simulator is treated as a black box function $f: \Phi \mapsto \mathbf{Y}$
where the burden of selecting appropriate outputs is typically left to the implementation of $f$. For more complex
simulators, however, the size and complexity of the outputs may require a more complex observation operator that
maps from the space of simulator states/outputs to observables. `SimulationBasedInference` provides a generic API
for handling simulation data that is consumed by [`Observables`](@ref "observables").

The simulation data storage API consists of three layers:

| Type | Role |
|------|------|
| [`StorageBackend`](@ref) | Represents the backing store for all simulation data; e.g. in memory vs on disk. |
| [`SimulationData`](@ref) | Lightweight view of the data for a single simulation stored in the backend. |
| [`SimulationDataSet`](@ref) | Lightweight view of the data for multiple simulations stored in the backend. |

In addition, [`DataBuffer`](@ref)s, provide array-like access to a single named output or scratch series
inside a `SimulationData`.

## Storage backends

```@docs; canonical = false
SimulationBasedInference.StorageBackend
```

The default [`InMemoryStorage`](@ref) backend keeps everything in RAM:

```@docs; canonical = false
SimulationBasedInference.InMemoryStorage
```

The in-memory backend is suitable for simulators with relatively small output sizes or those
where simulation data storage and postprocessing is already handled internally. However, in
some cases, we may want to define observables that process the simulation data efficiently
"online" (i.e. during the simulation) rather than accumulate everything into memory. For these
cases, a `DiskStorageBackend` can be used instead:

```@docs; canonical = false
SimulationBasedInference.DiskStorageBackend
```

Currently, `SimulationBasedInference` provides a single implementation of `DiskStorageBackend`
using distributed (per simulation) JLD2 files. This is implemented by the `SimulationBasedInferenceJLD2Ext`
extension module which is auto-loaded with `JLD2`:

```julia
using JLD2  # loads SimulationBasedInferenceJLD2Ext

storage = SimulationDataSet(
    backend = DiskStorageBackend(format"JLD2", "simulations/")
)
```

### StorageHandle

`StorageBackend`s provide an I/O handle API that keeps a connection open for the duration of a batch
of operations on a single simulation. Handles are obtained with `open(backend, index)` where `index` is
the index of the simulation.

```@docs; canonical = false
SimulationBasedInference.StorageHandle
```

The `open` implementation for backends supports a standard `do`-block syntax for automatically closing
the `StorageHandle` after use:

```julia
open(backend, 3) do h
    store_output!(h, :y, value)
    getinputs(h)
end  # handle closed automatically
```

Scratch storage, i.e. transient working buffers needed during a single simulation and accessed via
[`get_scratch_buffer`](@ref), are tied to the handle and are discarded once it is closed.

## SimulationData

`SimulationData` represents the main interface for reading and writing data for a single simulation.
It is passed to [`observe!`](@ref), [`initialize!`](@ref), and [`getvalue`](@ref) when computing or
reading observables during a forward solve.

```@docs; canonical = false
SimulationData
```

### Reading and writing

```@docs; canonical = false
store!(data::SimulationData, name::Symbol, value)
getoutput(data::SimulationData, name::Symbol)
getoutputs(data::SimulationData)
```

`getinputs(data)` and `getmetadata(data)` return the input parameters and metadata
dictionary stored for the simulation, respectively.

### Buffer access

For observable implementations that need to accumulate values over several simulator steps
(e.g. [`TimeSampled`](@ref)), `SimulationData` provides lazily-opened buffer views:

```@docs; canonical = false
SimulationBasedInference.get_output_buffer
SimulationBasedInference.get_scratch_buffer
```

The `with_output_buffer` and `with_scratch_buffer` helpers open a buffer, run a function,
and close the handle in a single call:

```@docs; canonical = false
SimulationBasedInference.with_output_buffer
SimulationBasedInference.with_scratch_buffer
```

### Lifecycle

```@docs; canonical = false
Base.close(data::SimulationData)
```

`Base.empty!(data::SimulationData)` clears all output series stored for the simulation.

## SimulationDataSet

`SimulationDataSet` is the top-level container used throughout an inference run. It is passed
to `init` and accumulates one `SimulationData` entry per forward solve.

```@docs; canonical = false
SimulationDataSet
```

### Allocating and storing simulations

```@docs; canonical = false
allocate!(storage::SimulationDataSet, inputs=nothing; metadata...)
store!(storage::SimulationDataSet, data::SimulationData; metadata...)
```

### Indexing and iteration

`SimulationDataSet` supports standard Julia indexing and iteration:

```julia
# index — returns a SimulationData view
sim = storage[3]

# iteration — yields (inputs, outputs, metadata) triples
for (inputs, outputs, metadata) in storage
    @show inputs, metadata[:iter]
end

# number of completed simulations
length(storage)
```

### Querying across iterations

```@docs; canonical = false
iterations(storage::SimulationDataSet)
```

## DataBuffer

`DataBuffer` is a low-level array-like view of a single named series (output or scratch)
inside a `SimulationData`. It is primarily used internally by observable implementations.

```@docs; canonical = false
SimulationBasedInference.DataBuffer
```

`DataBuffer` supports the standard Julia `push!`, `getindex`, `length`, `empty!`, `collect`,
`first`, `last`, and `iterate` methods, all forwarding through the underlying
[`StorageHandle`](@ref).

## Custom backend implementations

New `StorageBackend` subtypes must implement the following methods (dispatching on both the
backend directly for the slow path, and on a concrete `StorageHandle` subtype for the fast
path):

| Method | Description |
|--------|-------------|
| `open(backend, index) -> StorageHandle` | Open handle for simulation `index` |
| `allocate!(handle, input; metadata...)` | Allocate a new simulation slot |
| `num_simulations(backend)` | Total number of stored simulations |
| `getinputs(handle)` / `setinputs!(handle, x)` | Read/write inputs |
| `getmetadata(handle)` / `setmetadata!(handle; kwargs...)` | Read/write metadata |
| `ensure_output!(handle, name)` | Create an output series if absent |
| `store_output!(handle, name, x)` | Append to an output series |
| `get_output(handle, name, j)` | Retrieve the `j`-th element of an output series |
| `get_outputs(handle, name)` | Retrieve the full output series as a `Vector` |
| `output_length(handle, name)` | Length of an output series |
| `output_names(handle)` | Names of all output series |
| `has_output(handle, name)` | Check if an output series exists |
| `empty_output!(handle, name)` | Clear an output series |
| `Base.close(handle)` | Close the handle and discard scratch |
| `Base.isopen(handle)` | Whether the handle is still open |

Scratch storage is handled by default through the `h.scratch::Dict{Symbol,Any}` field
on the handle (see `StorageHandle`); backends can override it if needed.
