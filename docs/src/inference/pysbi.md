# Simulation-based inference via `sbi`

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference
    using SimulationBasedInference.PySBI

    import CommonSolve: solve
end
```

## Overview

Modern methods for simulation-based inference (SBI) leverage powerful and expressive *density estimators*, often parameterized by neural networks. These algorithms are designed to solve inference problems of the form:

$$
p(\theta \mid y) \propto p(\theta)\int_\mathcal{Z} p(y \mid z, \theta)p(z\mid\theta)dz
$$

where $z$ corresponds to some set of latent variables internal to the simulator. In the context of dynamical systems, $z$ often represents the trajectory of the system state(s) over time.

In cases where $z$ is a random variable, e.g., in the presence of stochastic dynamics or model errors, obtaining numerical estimates of the posterior distribution $p(\theta \mid y)$ formally requires integrating over all possible latent states of the simulator in the likelihood, which is usually intractable. This precludes the use of most standard methods for statistical inference.

Traditionally, this problem was solved using so-called [Approximate Bayesian Computation](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation) (ABC) methods which sidestep the need to compute the likelihood by simplying comparing low-dimensional summary statistics computed from the simulator outputs. More modern methods use powerful *density estimation* techniques to approximate the likelihood or the posterior directly. The python package [sbi](https://sbi-dev.github.io/sbi/0.22/) provides an easy-to-use interface for applying such methods to a wide variety of problems.

`SimulationBasedInference.jl` provides a Julia-wrapper around `sbi` via the `PySBI` extension module, which is loaded using `Requires.jl`. In order for `PySBI` to be loaded, the [`PythonCall`](https://github.com/JuliaPy/PythonCall.jl) package must be installed in your local environment.

Once `PythonCall` is installed, the `PySBI` module can be imported as

```julia
import PythonCall
import SimulationBasedInference.PySBI
```

The python module `sbi` can be directly accessed via `PySBI.sbi`.

The [PySNE](@ref) type wraps an `sbi` algorithm (by default `NPE_C`) which can then be used with the standard `CommonSolve` interface:

```docs
init(::SimulatorInferenceProblem, ::PySNE)
```

## PySBI API reference

```@autodocs
Modules = [PySBI]
Order = [:type, :function]
```
