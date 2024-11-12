# [Observables](@id observables)

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference: Observable, SimulatorObservable, Transient, TimeSampled
end
```

Observables define outputs or derived quantities produced by the simulator which are of interest to the user, either for the purpose of inference (i.e. conditioning on data), validation, or debugging.

The observables API is one of the key features of `SimulationBasedInference` that separates the package from other software tools for automated Bayesian inference.
Many such tools consider only models of the form $y = f(\theta)$ where $\theta$ are the uknowns to be inferred and $y$ are the outputs of the simulator $f$ to be compared to data.
While mathematically convenient, this formulation ignores some basic practial realities of many scientific and engineering use cases:

1. The simulator may be computaionally non-trivial to evaluate, i.e. $f(\theta)$ may require significant computational resources to evaluate. 
2. The output of the simulator is often itself large and complex, e.g. a solution to a set of partial differential equations.
3. The output of the simulator is not always directly comparable to the available data.

In light of this, `SimulationBasedInference` formally treats the simulator inference pipeline as a composition of two parts:

```math
\begin{align*}
u &= f(\theta) \\
y_i &= g_i(u)
\end{align*}
```

where $u$ is the output or latent state of the simulator and $g_i$ for $i=1\dots m$ are $m$ quantities derived from $u$. More importantly, for the case where $f$ is a dynamical system defined by
an `ODEProblem` or similar and the solution $u(t)$ is a function over time (and possibly space), observables $g_i$ can be defined as functionals which are evaluated during each simulation thereby circumventing
the need to store the solution `u` explicitly.

## Basic usage

Observables are constructed using the `SimulatorObservable` type which consists of the following components:

1. A name/identifier represented as a `Symbol` (e.g. `:y1`)
2. A function of the form $y_i = g_i(u)$ where $u$ is the (possibly transient) state of the simulator and $y_i$ is the output.
3. An output type which defines the behavior of the observable and how the outputs are stored.
4. A set of coordinates for $y_i$ which define its shape and indices.

Currently, `SimulationBasedInference` provides two types of observable outputs:

- [Transient](@ref) output types (with alias `TransientObservable`) which naively evaluate `g_i` and store a pointer to the return value. Repeated evaluations of the observable overwrite previous values.
- [TimeSampled](@ref) output types (with alias `TimeSampledObservable`) for dynamical systems which define a sampling frequency and a set of save times. Outputs of `g_i` at each sample time are buffered and then aggregated at each save time according to some reducer function (e.g. `mean`, `sum`, etc.).

Type specific dispatches of the `SimulatorObservable` constructor for each output type as a convenience.

```@docs; canonical=false
SimulatorObservable(name::Symbol, f, coords::Tuple)
```

where `coords` are a `Tuple` of dimension sizes (e.g. `(2,)` for a 2-dimensional output) or coordinates (e.g. `(1:2,)`) that index the dimensions of the output. Dimensions from `DimensionalData` can also be directly supplied (e.g. `(X(1:10),Y(1:10))`). Note that the output of `g_i` must match the shape specified by `output_dims`.

For time sampled observables of dynamical systems, it is necessary to additionally provide an initial time stamp `t0` and a vector of timestamps `ts` which specify the time points at which the the buffered observable values should be saved.

```@docs; canonical=false
SimulatorObservable(
    name::Symbol,
    obsfunc,
    t0::tType,
    tsave::AbstractVector{tType},
    output_shape_or_coords::Tuple;
    reducer=mean,
    samplerate=default_sample_rate(tsave),
) where {tType}
```

Here is a simple example for a generic `ODEProblem` from the `SciML` package ecosystem:

```julia
prob = ODEProblem(dudt, u0  tspan)
output_dims = size(u0)
t0 = tspan[1]
ts = LinRange(tspan[1], tspan[2], 100)
# this observable will simply save the full state at each save point in `ts`
obs = SimulatorObservable(:yt, state -> state.u, t0, ts, output_dims)
```

For time sampled observables, `output_dims` represents the shape of the output **at each time step**, i.e. without the time dimension which is imlicitly defined by the number of save points.

## Defining custom observables

In general, it is recommended to define custom observables using one of the pre-defined output types of `SimulatorObservable` described above.

However, in some cases, it may be easier to define a new observable type that is tailored to custom-tailored to a particular simulator or use case.

```@docs
SimulationBasedInference.Observable
```

This can be done by extending `SimulatorObservable` with a new `SimulatorOutput` type, or by defining an entirely new subtype of [Observable](@ref). Generally the former is recommended.

In either case, subtypes of `Observable` must implement the following methods.

```@docs; canonical=false
initialize!(obs::Observable, state)
```

```@docs; canonical=false
observe!(obs::Observable, state)
```

```@docs; canonical=false
getvalue(obs::Observable)
```

```@docs; canonical=false
setvalue!(obs::Observable, value)
```

```@docs; canonical=false
coordinates(obs::Observable)
```

New subtypes of `Observable` should additionally implement [Base.nameof](@ref) to return a `Symbol` corresponding to the unique identifier of the observable.
