"""
    EmulatedObservables{names} <: SciMLBase.AbstractSciMLAlgorithm

Represents an emulated forward solver for observables `names` in a given forward problem.
The dimensionality of the output spaces of the emulators are assumed to match those of the
observables exactly.
"""
struct EmulatedObservables{names,emTypes<:Tuple{Vararg{Emulator}},predType} <: SciMLBase.AbstractSciMLAlgorithm
    emulators::NamedTuple{names,emTypes}
    predict_func::predType
end

# convenience constructor
EmulatedObservables(predict_func=predict; named_emulators...) = EmulatedObservables((; named_emulators...), predict_func)

mutable struct EmulatedObservablesSolver
    forward_prob::SimulatorForwardProblem
    emobs::EmulatedObservables
    names::Tuple{Vararg{Symbol}}
    preds::Any
end

function CommonSolve.init(
    forward_prob::SimulatorForwardProblem,
    emobs::EmulatedObservables{names};
    p=forward_prob.prob.p,
    kwargs...
) where {names}
    obs_names = keys(forward_prob.observables)
    # take the intersection of the observables defined on forward problem and emulators
    em_names = filter(∈(obs_names), names)
    # check that the resulting intersection is not empty
    @assert !isempty(em_names) "None of $names found in observables $obs_names"
    return EmulatedObservablesSolver(remake(forward_prob, p=p, copy_observables=false), emobs, Tuple(names), missing)
end

function CommonSolve.step!(solver::EmulatedObservablesSolver)
    forward_prob = solver.forward_prob
    emulators = solver.emobs.emulators
    # evaluate emulator for each specified observable
    preds = map(nm -> solver.emobs.predict_func(emulators[nm], forward_prob.prob.p), solver.names)
    solver.preds = NamedTuple{solver.names}(preds)
    for name in solver.names
        obs = forward_prob.observables[name]
        # reshape to match shape of observable
        em_preds = reshape(solver.preds[name], size(obs))
        # update observable state
        SBI.setvalue!(obs, em_preds)
    end
    return false
end

function CommonSolve.solve!(solver::EmulatedObservablesSolver)
    step!(solver)
    return SimulatorForwardSolution(solver.forward_prob, solver.preds)
end
