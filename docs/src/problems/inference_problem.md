# [Inference problems](@id inference)

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference, OrdinaryDiffEq, ComponentArrays
end
```

The [SimulatorInferenceProblem](@ref) defines a probabilistic inverse problem from a given [SimulatorForwardProblem](@ref), a solver for that forward problem, a prior for the simulator parameters,
and a set of `SimulatorLikelihood`s which relate the output of one or more observables to the data on which we wish to condition. In other words, the `SimulatorInferenceProblem` fully specifies
the probability model,

$$
p(\phi,\sigma_1,\dots,\sigma_n | y_1,\dots,y_n,\mathcal{M}) \propto p(\phi|\mathcal{M})\prod_{i=1}^n p(\sigma_i) \prod_{i=1}^n p(y_i|\phi,\sigma_i,\mathcal{M})
$$
for some forward model $\mathcal{M}$ with parameters $\phi$ and a set of observables mapping to data $y_i$ each with noise model parameters $\sigma_i$.

```@docs; canonical=false
SimulatorInferenceProblem(
    prob::SimulatorForwardProblem,
    forward_solver,
    prior::AbstractSimulatorPrior,
    likelihoods::SimulatorLikelihood...;
    metadata::Dict=Dict(),
)
```

## Likelihoods (or the lack thereof)

Each [SimulatorLikelihood](@ref) defines a *likelihood* or *predictive distribution* of some data given an observable defined for the simulator.

```@docs; canonical=false
SimulatorLikelihood(::Type{distType}, obs, data, prior, name=nameof(obs)) where {distType}
```

Note that the `prior` passed to the `SimulatorLikelihood` is the prior for the observation noise/error model, not the forward model. To illustrate this distinction, consisder as an example a simple classical linear regression:

$$
\begin{align*}
y_i &\sim \mathcal{N}(\mu, \sigma) \\
\mu &= \beta x
\beta &\sim \mathcal{N}(0,1)
\sigma &\sim \text{InvGamma}(2,3)
\end{align*}
$$

In this case, the forward model parameters $\phi = \beta$ whereas the noise model parameters for the likelihood correspond to $\sigma$. Note that `SimulatorInferenceProblem` always assumes that $\sigma$ is *a priori* independent of $\phi$!

Currently, `SimulatoionBasedInference` defines the following types of likelihoods:

```@docs; canonical=false
GaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=Exponential(1.0)),
    name,
)
```

```@docs; canonical=false
IsotropicGaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=Exponential(1.0)),
    name,
)
```

```@docs; canonical=false
DiagonalGaussianLikelihood(
    obs,
    data,
    prior=NamedProductPrior(σ=filldist(Exponential(1.0), prod(size(obs)))),
    name,
)
```

```@docs; canonical=false
DiracLikelihood(
    obs,
    data,
    name,
)
```

### Likelihood "free" inference

One common use case of simulation-based inference is for so-called "likelihood free" models, i.e. models where the likelihood is either intractable or not well defined. For these cases, `SimulationBasedInference` provides a special type:

```@docs; canonical=false
ImplicitLikelihood(
    obs,
    data,
    name,
)
```

The implicit liklihood is virtually identiical to `DiracLikelihood` in that the observable is compared directly to the given data. However, unlike `DiracLikelihood` the `ImplicitLikelihood` does not allow construction of a predictive distribution
or evaluation of the log-liklihood since they are not defined.

### Custom likelihoods

Currently, all likelihoods must be defined using the `SimulatorLikelihood` type. However, new dispatches may be defined for `SimulatorLikelihood`s of for new distribution families (i.e. `distType`) or with specific types of priors.

All `SimulatorLikelihood`s must implement the following method interface:

```@docs; canonical=false
observable(lik::SimulatorLikelihood)::SimulatorObservable
```

```@docs; canonical=false
getprior(lik::SimulatorLikelihood)
```

```@docs; canonical=false
predictive_distribution(lik::SimulatorLikelihood, args...)
```

```@docs; canonical=false
sample_prediction([rng::AbstractRNG], lik::SimulatorLikelihood, args...)
```

## Parameter spaces

All `SimulatorInferenceProblem`s implicitly define a coupling between three parameter spaces: an unconstrained sample space $\Xi$, a constrained sample space $\Theta$, and the model parameter space $\Phi$.
The "sample space" here refers to the parameter space defined by the *joint prior* distribution, i.e. the product of the model parameter prior and the noise model prior $p(\phi,\sigma) = p(\phi)p(\sigma)$.
This is the parameter space of interest for the inference problem, i.e. for optimization or posterior sampling. There is a noteworthy distinction made here between the sample space of the model parameters
and the actual model parameter space. This distinction arises for three primary reasons:
1. Not all parameters defined by the simulator are necessarily included in the inference parameters. There may be some paraemters which are treated as constants because they are not of interest.
2. It is often advantageous or convenient to reparameterize some parameters; e.g. parameters that have linear dependencies on each other can be reparameterized as linearly independent parameters in the prior.
3. Hierarchical models may define additional parameters which are conditioned on by each model parameter.

We can express this more formally in terms of the above probability model as,
$$
p(\phi | y,\pi) \propto p(y|\phi,\pi)p(\phi|\theta,\pi)p(\theta|\pi)
$$
where $\phi = \pi(\theta)$ can be referred to as the *prior model* since it defines the statistical relationship between the sampled parameters $\theta$ and the "real" model parameters $\phi$. Note that the
conditioning on the forward model $\mathcal{M}$ is here suppressed for brevity. In cases where $\pi$ is a deterministic mapping, the additional density terms $p(\phi|\theta)$ and $p(\theta)$ can be safely
neglected.

The *unconstrained* parameters $\xi \in \Xi$ represent a (typically nonlinear) bijection $\theta = f(\xi)$ of the sample space $\Theta$ onto the real numbers $\Xi = \mathbb{R}^d$ where $d$ is the number of parameters.
This bijection is often necessary in order to avoid problems with bounded domains in many sampling and optimization algorithms. Within the context of `SimulationBasedInfernece`, we refer to `f` as the *bijetor*,
$\pi$ as the *forward map*, and $\pi \circ f$ as the *unconstrained forward map*. Note that other authors sometimes use the term "forward map" to additionally include the forward model $\mathcal{M}$. We instead
refer to the full forward mapping as the *joint forward model*, i.e. $\mathcal{M} \circ \pi : \Theta \mapsto \mathbf{Y}$ where $y \in \mathbf{Y}$.
