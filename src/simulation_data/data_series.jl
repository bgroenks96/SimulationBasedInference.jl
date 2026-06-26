"""
    DataSeries{kind,B<:StorageBackend}

A lightweight, ordered, appendable **view** onto a single named series of values living in a
[`StorageBackend`](@ref). `kind` is either `:output` (a persistent observable output series)
or `:scratch` (a transient working buffer); it selects which backend methods the view
forwards to. A `DataSeries` owns no data of its own — `store!`, `getindex`, `length`, and
`clear!` forward to the backend. This is the interface that observables read from and write
to (via `getbuffer`, `make_buffer!`, `get_buffer`).

The remaining methods (`push!`, `empty!`, `iterate`, `collect`, `first`/`last`, ...) are
provided generically in terms of the four forwarded primitives.
"""
struct DataSeries{kind,B<:StorageBackend}
    backend::B
    sim::Int
    name::Symbol
end

DataSeries{kind}(backend::B, sim::Integer, name::Symbol) where {kind,B<:StorageBackend} =
    DataSeries{kind,B}(backend, Int(sim), name)

# forwarded primitives — :output
store!(s::DataSeries{:output}, x) = (store_output!(s.backend, s.sim, s.name, x); s)
Base.getindex(s::DataSeries{:output}, i::Integer) = get_output(s.backend, s.sim, s.name, i)
Base.length(s::DataSeries{:output}) = output_length(s.backend, s.sim, s.name)
clear!(s::DataSeries{:output}) = (clear_output!(s.backend, s.sim, s.name); s)

# forwarded primitives — :scratch
store!(s::DataSeries{:scratch}, x) = (store_scratch!(s.backend, s.sim, s.name, x); s)
Base.getindex(s::DataSeries{:scratch}, i::Integer) = get_scratch(s.backend, s.sim, s.name, i)
Base.length(s::DataSeries{:scratch}) = scratch_length(s.backend, s.sim, s.name)
clear!(s::DataSeries{:scratch}) = (clear_scratch!(s.backend, s.sim, s.name); s)

# generic derived methods (kind-agnostic)
Base.push!(s::DataSeries, x) = store!(s, x)
Base.empty!(s::DataSeries) = clear!(s)
Base.firstindex(::DataSeries) = 1
Base.lastindex(s::DataSeries) = length(s)
Base.isempty(s::DataSeries) = length(s) == 0
Base.size(s::DataSeries) = (length(s),)
Base.getindex(s::DataSeries, r::AbstractVector) = [s[i] for i in r]
Base.iterate(s::DataSeries, state::Int=1) = state <= length(s) ? (s[state], state + 1) : nothing
Base.collect(s::DataSeries) = [s[i] for i in 1:length(s)]
Base.first(s::DataSeries) = s[1]
Base.last(s::DataSeries) = s[length(s)]
flush!(s::DataSeries) = flush!(s.backend)
Base.close(::DataSeries) = nothing
