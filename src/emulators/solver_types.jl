"""
    EmulatedObservables{names}

Represents an emulated forward solver for observables `names` in a given forward problem.
The dimensionality of the output spaces of the emulators are assumed to match those of the
observables exactly.
"""
struct EmulatedObservables{names}
    emulators::NamedTuple{names,Tuple{Vararg{Emulator}}}
end

# convenience constructor
EmulatedObservables(;named_emulators...) = EmulatedObservables((; named_emulators...))

"""
Evaluates the emulated forward pass of `forward_prob` using the given emulators, which are
assumed to be already initialized and/or trained.
"""
function CommonSolve.solve(
    forward_prob::SimulatorForwardProblem,
    emobs::EmulatedObservables{names};
    p=forward_prob.prob.p,
    predict_func=MMI.predict,
    kwargs...
) where {names}
    obs_names = keys(forward_prob.observables)
    # take the intersection of the observables defined on forward problem and emulators
    em_names = names ∩ obs_names
    # check that the resulting intersection is not empty
    @assert !isempty(em_names) "None of $names found in observables $obs_names"
    emulators = emobs.emulators
    # evaluate emulator for each specified observable
    preds = map(nm -> predict_func(emulators[nm], p), tuple(em_names...))
    for name in em_names
        obs = forward_prob.observables[name]
        # reshape to match shape of observable
        em_preds = reshape(preds[name], size(obs))
        # update observable state
        SBI.setvalue!(obs, em_preds)
    end
    return SimulatorForwardSolution(forward_prob, preds)
end
