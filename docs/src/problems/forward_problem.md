# [Forward problems](@id forward)

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference, OrdinaryDiffEq, ComponentArrays
end
```

The [SimulatorForwardProblem](@ref) defines a forward map $\mathcal{F}: \Phi \mapsto \{\mathbf{Y}_i\}$ for $i=1\dots N$ from parameters $\Phi$ to a set of $N$ [observables](@ref observables). Observables are functions of the simulator state that can be either diagnostics or true observable quantities that may be later compared to data.

A `SimulatorForwardProblem` is a wrapper around any other type of [SciML problem](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/) that characterizes a simulator as well as one or more observables computed from the output of the simulation.

```@docs; canonical=false
SimulatorForwardProblem(
    prob::SciMLBase.AbstractSciMLProblem,,
    observables::SimulatorObservable...
)
```

The simulator can also be defined by any arbitrary Julia function of the form `f(::AbstractVector)::Any` in which case the function is treated as a simple forward map from input parameters to some output type that is processed by the `Observable`s.

```@docs; canonical=false
SimulatorForwardProblem(
    f,
    p0::AbstractVector,
    observables::SimulatorObservable...
)
```

Note that forward problems typically need to be defined with some initial input parameters:
```@example
p0 = ones(10)
forward_prob = SimulatorForwardProblem(sum, p0)
```

These parameters can be updated using the `remake` method from `SciMLBase`:
```@example
new_prob = remake(forward_prob, p=2*p0)
```

## Solve interface

Like any SciML problem, `SimulatorForwardProblem` supports the [SciML solve](https://docs.sciml.ai/SciMLBase/stable/interfaces/Init_Solve/) interface which allows the forward problem to be solved using any appropriate solver for the underlying problem.

For example, a ODE-based simulator can be solved using algorithms from [OrdinaryDiffEq](https://docs.sciml.ai/OrdinaryDiffEq/stable/):

```@example
p = ComponentArray(α=0.1)
odeprob = ODEProblem((u,p,t) -> -p.α*u, [1.0], (0.0,1.0), p)
observable = SimulatorObservable(:u, state -> state.u, 0.0, 0.1:0.1:1.0, size(odeprob.u0), samplerate=0.01)
forward_prob = SimulatorForwardProblem(odeprob, observable)
forward_sol = solve(forward_prob, Tsit5())
```

The return value of `solve` for `SimulatorForwardProblem` is typically a [SimulatorForwardSolution](@ref) which wraps both the underlying solution type as well as the original forward problem and its corresponding observables.

For forward problems such as the ODE example above that involve iteration, the problem can also be solved iteratively using `init` and `step!`:

```@example
solver = init(forward_prob)
step!(solver) # one solver step
for i in solver
    # iterate until finished
end

# alternatively, solve!
sol = solve!(solver)
```

