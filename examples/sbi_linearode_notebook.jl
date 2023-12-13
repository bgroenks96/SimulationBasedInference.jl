### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# ╔═╡ 110c7946-9908-11ee-16b3-b35854cd4878
begin
	import Pkg
	Pkg.activate(dirname(Base.current_project()))
end

# ╔═╡ 5fea4149-21e4-47d4-aed5-b1bc2dae7968
using SimulationBasedInference

# ╔═╡ d18a60ea-59eb-4d35-a410-2ecb7c57d48c
using Plots

# ╔═╡ ab868ec0-4c11-4855-8ea0-2da796f7b13c
begin

using OrdinaryDiffEq
using Statistics

import Random

function linear_ode_problem(
    α_true=0.2;
    σ_y=0.01,
    α_prior=Beta(1,1),
    σ_prior=Exponential(σ_y),
    ode_solver=Tsit5(),
	tspan=(0.0,10.0),
    rng=Random.default_rng(),
)
    # set up linear ODE problem with one parameter
    ode_p = ComponentArray(α=α_true)
    odeprob = ODEProblem((u,p,t) -> -p[1]*u, [1.0], tspan, ode_p)
	tsave = tspan[1]+0.1:0.1:tspan[end]
	n_obs = length(tsave)
    # observable extracts state from integrator
    observable = SimulatorObservable(:obs, integrator -> integrator.u, 0.0, tsave, samplerate=0.01)
    # specify forward problem
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    # generate "true" solution
    forward_sol = solve(forwardprob, ode_solver)
    @assert forward_sol.sol.retcode == ReturnCode.Default
    true_obs = retrieve(observable)
    # specify priors
    prior = PriorDistribution(α=α_prior)
    noise_scale = σ_y
    noise_scale_prior = PriorDistribution(:σ, σ_prior)
    # create noisy data
    noisy_obs = true_obs .+ noise_scale*randn(rng, n_obs)
    # simple Gaussian likelihood; note that we're cheating a bit here since we know the noise level a priori
    lik = SimulatorLikelihood(IsoNormal, observable, noisy_obs, noise_scale_prior)
    # inference problem
    inference_prob = SimulatorInferenceProblem(forwardprob, ode_solver, prior, lik)
    return inference_prob
end

end

# ╔═╡ f7e3f79f-b7c5-427f-a86b-7813b573fb57
using SimulationBasedInference.Ensembles

# ╔═╡ e7a372a6-ee3c-40f2-9a19-9bf7c0409713
html"""<style>
main {
	max-width: 70%;
	margin-right: 0;
}
"""

# ╔═╡ c4b00171-c64d-4d21-bf9b-76359f56b4e9
md"""
# SimulationBasedInference.jl: A new computational framework for simulation-based inference in Julia

#### Author: Brian Groenke
#### December 2023
"""

# ╔═╡ 8875aa4b-d9a9-4859-a356-8c6fcc424e9d
md"""
## Introduction
Simulator-type models are ubiquitous in science and engineering.

Most (all?) models require some kind of input, e.g boundary conditions (forcings), physical properties or constants, etc.

Often, these parameters are not fully known *a priori*... but usually we know something!
"""

# ╔═╡ 56049286-9ef8-4544-b930-885c3f74245a
md"""
Bayesian inference provides a natural framework for constraining this uncertainty using observed data:

```math
p(\theta | \mathbf{y}) = \frac{p(\mathbf{y}|\theta)p(\theta)}{p(\mathbf{y})}
```

The **posterior distribution** $p(\theta | \mathbf{y})$ represents our **best estimate** (with uncertainty) of the unknown parameters $\theta$ after observing $\mathbf{y}$.
"""

# ╔═╡ 4827d185-7859-42bb-bef0-3b23aa9c15c2
md"""
## Simulation-based inference

Simulation-bsaed inference (SBI) refers to the problem of performing **statistical inference** (Bayesian or otherwise) of unknown parameters $\theta$ where the forward model $\mathcal{M}$:

```math
y = \mathcal{M}(\theta) + \epsilon
```

is a dynamical model or physics-based *simulator*.

In the numerical modeling literature, this is often referred to as *data assimilation*.

There are two fundamental challenges with this problem:
1. The model $\mathcal{M}$ is almost always *non-linear* and, in the case of dynamical models, *intractable* (i.e. we cannot write down the solution a priori).
2. Evaluating the forward map $\mathcal{M}(\theta)$ is usually non-trivial, i.e. **computationally expensive** or at least inconvenient.

Thus, classical statistical methods that rely on either analytical or numerical methods to derive the posterior distribution are generally difficult (or impossible) to apply.
"""

# ╔═╡ df2ae8ba-8904-49e5-81b5-09408fc21d7c
md"""
"""

# ╔═╡ 02c02e64-21ef-4bab-b497-440405b2ef0a
inference_prob = linear_ode_problem();

# ╔═╡ b1fbaab7-cf8c-4aee-8298-666718112cf8
forward_sol = solve(inference_prob.forward_prob, Tsit5(), saveat=0.1);

# ╔═╡ c2efee16-04de-4c5d-848c-5e0346302cf9
plot(forward_sol.sol, label="True solution")

# ╔═╡ bbd6aa72-e1c1-4eb7-b887-8937f758af2b
pbs_sol = solve(inference_prob, PBS())

# ╔═╡ 006fbbd2-fec1-489a-93c0-b047f7310588
posterior_mean_pbs = getensemble(pbs_sol.result)*get_weights(pbs_sol.result)

# ╔═╡ 5cf85642-e594-41ae-8e2c-e2403f67d128
plot([reduce(vcat, forward_sol.sol.u[2:end]), pbs_sol.result.predictions*get_weights(pbs_sol.result)])

# ╔═╡ c957e3cc-bfc4-4d0c-b0cf-fad1dee38bba
esmda_sol = solve(inference_prob, ESMDA())

# ╔═╡ 87d62425-2d14-4643-b039-b2f1258a1b0a
eks_sol = solve(inference_prob, EKS(), verbose=false)

# ╔═╡ Cell order:
# ╟─e7a372a6-ee3c-40f2-9a19-9bf7c0409713
# ╟─110c7946-9908-11ee-16b3-b35854cd4878
# ╟─c4b00171-c64d-4d21-bf9b-76359f56b4e9
# ╠═8875aa4b-d9a9-4859-a356-8c6fcc424e9d
# ╠═56049286-9ef8-4544-b930-885c3f74245a
# ╠═4827d185-7859-42bb-bef0-3b23aa9c15c2
# ╠═df2ae8ba-8904-49e5-81b5-09408fc21d7c
# ╠═5fea4149-21e4-47d4-aed5-b1bc2dae7968
# ╠═d18a60ea-59eb-4d35-a410-2ecb7c57d48c
# ╠═ab868ec0-4c11-4855-8ea0-2da796f7b13c
# ╠═02c02e64-21ef-4bab-b497-440405b2ef0a
# ╠═b1fbaab7-cf8c-4aee-8298-666718112cf8
# ╠═c2efee16-04de-4c5d-848c-5e0346302cf9
# ╠═f7e3f79f-b7c5-427f-a86b-7813b573fb57
# ╠═bbd6aa72-e1c1-4eb7-b887-8937f758af2b
# ╠═006fbbd2-fec1-489a-93c0-b047f7310588
# ╠═5cf85642-e594-41ae-8e2c-e2403f67d128
# ╠═c957e3cc-bfc4-4d0c-b0cf-fad1dee38bba
# ╠═87d62425-2d14-4643-b039-b2f1258a1b0a
