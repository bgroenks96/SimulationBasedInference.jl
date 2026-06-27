"""
    DataStore{kind,B<:StorageBackend}

A lightweight, ordered, appendable **view** onto a single named series of values living in a
[`StorageBackend`](@ref). `kind` is either `:output` (a persistent observable output series)
or `:scratch` (a transient working buffer); it selects which backend methods the view
forwards to. A `DataStore` owns no data of its own — `store!`, `getindex`, `length`, and
`empty!` forward to the backend. This is the interface that observables read from and write
to (via `get_output_storage`, `create_scratch!`, `get_scratch_storage`).

The remaining methods (`push!`, `empty!`, `iterate`, `collect`, `first`/`last`, ...) are
provided generically in terms of the four forwarded primitives.
"""
struct DataStore{kind,B<:StorageBackend}
    backend::B
    sim::Int
    name::Symbol
end

DataStore{kind}(backend::B, sim::Integer, name::Symbol) where {kind,B<:StorageBackend} =
    DataStore{kind,B}(backend, Int(sim), name)

# forwarded primitives — :output
store!(s::DataStore{:output}, x) = (store_output!(s.backend, s.sim, s.name, x); s)
Base.getindex(s::DataStore{:output}, i::Integer) = get_output(s.backend, s.sim, s.name, i)
Base.length(s::DataStore{:output}) = output_length(s.backend, s.sim, s.name)
Base.empty!(s::DataStore{:output}) = empty_output!(s.backend, s.sim, s.name)

# forwarded primitives — :scratch
store!(s::DataStore{:scratch}, x) = (store_scratch!(s.backend, s.sim, s.name, x); s)
Base.getindex(s::DataStore{:scratch}, i::Integer) = get_scratch_storage(s.backend, s.sim, s.name, i)
Base.length(s::DataStore{:scratch}) = scratch_length(s.backend, s.sim, s.name)
Base.empty!(s::DataStore{:scratch}) = empty_scratch!(s.backend, s.sim, s.name)

# generic derived methods (kind-agnostic)
Base.push!(s::DataStore, x) = store!(s, x)
Base.firstindex(::DataStore) = 1
Base.lastindex(s::DataStore) = length(s)
Base.isempty(s::DataStore) = length(s) == 0
Base.size(s::DataStore) = (length(s),)
Base.getindex(s::DataStore, r::AbstractVector) = [s[i] for i in r]
Base.iterate(s::DataStore, state::Int=1) = state <= length(s) ? (s[state], state + 1) : nothing
Base.collect(s::DataStore) = [s[i] for i in 1:length(s)]
Base.first(s::DataStore) = s[1]
Base.last(s::DataStore) = s[length(s)]
