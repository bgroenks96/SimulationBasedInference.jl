using Gen: VectorTraceChoiceMap, StaticChoiceMap

to_static_choice_map(other::StaticChoiceMap) = other
function to_static_choice_map(other::DynamicChoiceMap)
    leaf_keys_and_nodes = collect(get_values_shallow(other))
    internal_keys_and_nodes = collect(get_submaps_shallow(other))
    if length(leaf_keys_and_nodes) > 0
        (leaf_keys, leaf_nodes) = collect(zip(leaf_keys_and_nodes...))
    else
        (leaf_keys, leaf_nodes) = ((), ())
    end
    if length(internal_keys_and_nodes) > 0
        (internal_keys, internal_nodes) = collect(zip(internal_keys_and_nodes...))
    else
        (internal_keys, internal_nodes) = ((), ())
    end
    internal_nodes = map(to_static_choice_map, internal_nodes)
    @assert all(isa.(leaf_keys, Symbol)) "Composite keys are currently not supported"
    @assert all(isa.(internal_keys, Symbol)) "Composite keys are currently not supported"
    leaves = NamedTuple{leaf_keys}(leaf_nodes)
    internal = NamedTuple{internal_keys}(internal_nodes)
    StaticChoiceMap(
        leaves,
        internal,
        isempty(other),
    )
end

choicemap2nt(choices::ChoiceMap) = choicemap2nt(to_static_choice_map(choicemap(choices)))
function choicemap2nt(choices::StaticChoiceMap)
    nt_internal_nodes = map(choicemap2nt, choices.internal_nodes)
    return merge(choices.leaf_nodes, nt_internal_nodes)
end
