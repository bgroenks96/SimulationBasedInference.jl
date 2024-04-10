### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 110c7946-9908-11ee-16b3-b35854cd4878
begin
	import Pkg
	Pkg.activate(dirname(Base.current_project()))

	using PlutoUI
end

# ╔═╡ 5fea4149-21e4-47d4-aed5-b1bc2dae7968
using SimulationBasedInference

# ╔═╡ 4cce20ca-c370-4e5c-80d6-ff48091fe20b
using OrdinaryDiffEq

# ╔═╡ b543cd46-e098-4f90-90ed-4ed72fad7c01
using Plots, StatsPlots

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

We need solutions such as **ensemble** or **particle-based** sampling methods that work better for such use cases.
"""

# ╔═╡ a7dbf72f-2392-4ce4-aa12-fd6382621d03
md"""
**SimulationBasedInference.jl** is an upcoming Julia package that aims to centralize many of these methods under a single unifying framework for use in Earth system modeling.
"""

# ╔═╡ df2ae8ba-8904-49e5-81b5-09408fc21d7c
md"""
## Linear ODE: Forward problem setup
"""

# ╔═╡ e1509fe6-41d4-4186-b4a9-b98231862883
import Random

# ╔═╡ 018c6725-07d5-49a6-8af5-27b2aad46402
tspan = (0.0,10.0);

# ╔═╡ 6dd0a882-5398-4840-99fa-fd8ac840bc9e
rng = Random.MersenneTwister(1234);

# ╔═╡ 24a169a1-9f40-47d6-811a-cc85248c71e9
ode_func(u,p,t) = -p[1]*u;

# ╔═╡ 5e5488f8-12ba-41e2-b47c-c2b3e8937a3f
tsave = tspan[1]+0.1:0.1:tspan[end];

# ╔═╡ f0dd4069-0fb0-43cb-b144-afa5aea029d1
n_obs = length(tsave);

# ╔═╡ 348a8746-6a42-4820-8f2e-f1b16e005537
# observable extracts state from integrator
observable = SimulatorObservable(:y, integrator -> integrator.u, 0.0, tsave, samplerate=0.01);

# ╔═╡ 5b8fd3fd-8f9f-431e-9aff-0df3f26c5f6e
f(p) = sum(p)

# ╔═╡ 6d7efe46-4602-4282-97ef-4f0a56285589
SimulatorForwardProblem(f, [0.0])

# ╔═╡ b60cd64b-0308-4040-a68f-6ca106a88dcb
ode_solver = Tsit5();

# ╔═╡ 54ce6edd-eae9-4bab-90f9-6e05c5647c98
@bind α_true NumberField(0.0:0.01:1.0, default=0.2)

# ╔═╡ f2155b91-85bb-4da6-894c-1dc6a42f33c9
# set up linear ODE problem with one parameter
ode_p = [α_true];

# ╔═╡ c6cccee4-7a36-4fee-b619-92550f9bd16a
odeprob = ODEProblem(ode_func, [1.0], tspan, ode_p)

# ╔═╡ 175f41ae-2fe3-4b75-ae19-b6384334e490
# specify forward problem
forward_prob = SimulatorForwardProblem(odeprob, observable);

# ╔═╡ 13c53788-b4d6-41ad-9a02-d76c1cf8473c
# generate "true" solution
forward_sol = solve(forward_prob, ode_solver, saveat=0.01);

# ╔═╡ eda46bb2-e07c-4ec8-8959-f5e52c8d5202
true_obs = retrieve(forward_sol.prob.observables.y);

# ╔═╡ 945ab911-b996-4354-ba8a-7987e9bdbd45
@bind noise_scale NumberField(0.01:0.01:0.1, default=0.05)

# ╔═╡ 2cf9a8ab-a709-430f-9564-8ca409e0c4c9
# create noisy data
noisy_obs = true_obs .+ noise_scale*randn(rng, n_obs);

# ╔═╡ c2efee16-04de-4c5d-848c-5e0346302cf9
begin
	plot(forward_sol.sol, label="True solution", linewidth=3, color=:black)
	scatter!(tsave, noisy_obs, label="Noisy observations", alpha=0.5)
end

# ╔═╡ ff7551ce-531c-4322-8c9d-7204e8b7937d
md"""
## Inference (inverse) problem setup
"""

# ╔═╡ ec0c2989-c67d-4f33-85c9-5db53907a457
α_prior_dist = Beta(2,2);

# ╔═╡ e97336d9-f618-4c76-abac-6c934863834b
# note: we're cheating here by using the true noise scale as the mean!
noise_prior_dist = Exponential(noise_scale);

# ╔═╡ 019fe7bd-c001-4fa6-bef3-0f67cb49569c
# specify model prior
model_prior = prior(α=α_prior_dist);

# ╔═╡ dc3e728f-4af4-42ec-9d67-0a09c125b826
noise_scale_prior = prior(σ=noise_prior_dist);

# ╔═╡ 882edb35-5bd4-4248-bcfc-1b2dc899abb5
# simple Gaussian likelihood; note that we're cheating a bit here since we know the noise level a priori
lik = SimulatorLikelihood(IsoNormal, observable, noisy_obs, noise_scale_prior);

# ╔═╡ e7d9860d-6072-4d14-838a-145d61bfac83
# inference problem
inference_prob = SimulatorInferenceProblem(forward_prob, ode_solver, model_prior, lik);

# ╔═╡ d1c066fb-a42e-4010-9b4b-97c70f8e2931
StatsPlots.plot(α_prior_dist, label="α prior")

# ╔═╡ ba2d4dce-dae4-485f-b9ca-1cdc4c161cd3
md"""
## Particle batch smoother (PBS)
"""

# ╔═╡ bbd6aa72-e1c1-4eb7-b887-8937f758af2b
pbs_sol = solve(inference_prob, PBS(), ensemble_size=256, rng=rng)

# ╔═╡ 006fbbd2-fec1-489a-93c0-b047f7310588
posterior_weights = get_weights(pbs_sol.result)

# ╔═╡ ab585b3f-5a93-4ae9-ad53-f1558654d6ef
histogram(posterior_weights)

# ╔═╡ 64705b6e-8de5-451f-99b2-f6e21c40b2f6
pbs_sol.result

# ╔═╡ 5ad22545-a42e-4415-a22a-1821979f9645
posterior_mean_pbs = get_transformed_ensemble(pbs_sol)*posterior_weights

# ╔═╡ ff3f3a04-8240-4255-b7d7-d3ea91b035ed
posterior_std_pbs = std(get_transformed_ensemble(pbs_sol), weights(posterior_weights))

# ╔═╡ eb00ac7f-f436-476b-8a3c-ceb6f6ed7999
pbs_sol.outputs[end]*get_weights(pbs_sol.result);

# ╔═╡ 5cf85642-e594-41ae-8e2c-e2403f67d128
begin
	plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="Particle batch smoother")
	plot!(tsave, pbs_sol.outputs[end]*posterior_weights, label="Posterior mean", c=:blue, linestyle=:dash, ribbon=2*posterior_std_pbs, alpha=0.7, linewidth=2)
		scatter!(tsave, noisy_obs, label="Noisy observations", c=:orange)
end

# ╔═╡ 85fd1bdb-dd69-496e-905a-2c4359bfd6cc
md"""
## Ensemble Kalman Smoother (ES-MDA)
"""

# ╔═╡ c957e3cc-bfc4-4d0c-b0cf-fad1dee38bba
esmda_sol = solve(inference_prob, ESMDA(), ensemble_size=256, rng=rng)

# ╔═╡ 08e115c3-96a8-413a-85ec-6d62bfd4783a
posterior_esmda = get_transformed_ensemble(esmda_sol)

# ╔═╡ a870267e-21bb-43a2-95f9-9a027057c6c8
posterior_mean_esmda = mean(posterior_esmda, dims=2)

# ╔═╡ bc3966fd-12e6-4b4a-85f7-0cfcd934244c
posterior_std_esmda = std(posterior_esmda, dims=2)

# ╔═╡ cf39907e-83e2-4dfb-a4c4-cacbfdc4f75c
begin
	plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="ES-MDA")
	plot!(tsave, mean(esmda_sol.outputs[end], dims=2), label="Posterior mean", c=:blue, linestyle=:dash, ribbon=2*posterior_std_esmda, alpha=0.7, linewidth=2)
		scatter!(tsave, noisy_obs, label="Noisy observations", c=:orange)
end

# ╔═╡ 8241bd0e-d5de-41a6-ad9c-d92c447e9f75
md"""
## Ensemble Kalman Sampler (EKS)

*Garbuno-Inigo et al. 2020. SIAM. Interacting Langevin diffusions: Gradient structure and ensemble Kalman sampler.*
"""

# ╔═╡ 87d62425-2d14-4643-b039-b2f1258a1b0a
eks_sol = solve(inference_prob, EKS(), ensemble_size=256, rng=rng, verbose=false)

# ╔═╡ 7f98c3fd-1841-4d39-8ad4-2446bee6285f
posterior_eks = get_transformed_ensemble(eks_sol)

# ╔═╡ 61c1cb94-74ff-46e2-acc8-45ae27cdeb24
posterior_mean_eks = mean(posterior_eks, dims=2)

# ╔═╡ 4671521e-2df4-47d9-b2db-cf7f1fc3e86a
posterior_std_eks = std(posterior_eks, dims=2)

# ╔═╡ 2385f99a-b14b-41fb-9c83-d7463f1cf0d8
begin
	plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="EKS")
	plot!(tsave, mean(eks_sol.outputs[end], dims=2), label="Posterior mean", c=:blue, linestyle=:dash, ribbon=2*posterior_std_eks, alpha=0.7, linewidth=2)
		scatter!(tsave, noisy_obs, label="Noisy observations", c=:orange)
end

# ╔═╡ e0b6bf9c-1e97-4ae0-9db1-7bd3bc574515
md"""
## Intercopmarison of all algorithms
"""

# ╔═╡ c93de303-23c1-4485-b0be-c87b4ebaf6aa
begin
	plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="EKS")
	plot!(tsave, pbs_sol.outputs[end]*posterior_weights, label="Posterior mean (PBS)", linestyle=:dash, ribbon=2*posterior_std_pbs, alpha=0.7, linewidth=3)
	plot!(tsave, mean(esmda_sol.outputs[end], dims=2), label="Posterior mean (ES-MDA)", linestyle=:dash, ribbon=2*posterior_std_esmda, alpha=0.7, linewidth=3)
	plot!(tsave, mean(eks_sol.outputs[end], dims=2), label="Posterior mean (EKS)", linestyle=:dash, ribbon=2*posterior_std_eks, alpha=0.7, linewidth=3)
	scatter!(tsave, noisy_obs, label="Noisy observations", c=:orange)
end

# ╔═╡ 5ce219b6-850e-4a69-b78b-e17fc6c2b205
md"""
## Next steps

- More interesting (but still simple) test cases!
  - Degree-day snow melt model
  - Simple permafrost model (heat conduction only)
  - Surface energy balance model from CryoGrid
- Add more algorithms
  - Calibrate, Emulate, Sample (Cleary et al. 2020); already implemented!
  - Wrap python [sbi](https://sbi-dev.github.io/sbi/) package which implements **neural density estimators** for SBI.
- Provide convenience wrappers for calling external model code (e.g. C, FORTRAN, python)
"""

# ╔═╡ 08453927-b13b-4250-a25f-31911341f8bb
md"""
## Thank you!

### Contact: brian.groenke@awi.de

### https://github.com/bgroenks96/SimulationBasedInference.jl
"""

# ╔═╡ Cell order:
# ╟─e7a372a6-ee3c-40f2-9a19-9bf7c0409713
# ╟─110c7946-9908-11ee-16b3-b35854cd4878
# ╟─c4b00171-c64d-4d21-bf9b-76359f56b4e9
# ╟─8875aa4b-d9a9-4859-a356-8c6fcc424e9d
# ╟─56049286-9ef8-4544-b930-885c3f74245a
# ╟─4827d185-7859-42bb-bef0-3b23aa9c15c2
# ╟─a7dbf72f-2392-4ce4-aa12-fd6382621d03
# ╟─df2ae8ba-8904-49e5-81b5-09408fc21d7c
# ╠═5fea4149-21e4-47d4-aed5-b1bc2dae7968
# ╠═2d1a8391-695a-425d-9ebd-01557b10200d
# ╠═4cce20ca-c370-4e5c-80d6-ff48091fe20b
# ╠═b543cd46-e098-4f90-90ed-4ed72fad7c01
# ╠═e1509fe6-41d4-4186-b4a9-b98231862883
# ╠═018c6725-07d5-49a6-8af5-27b2aad46402
# ╠═6dd0a882-5398-4840-99fa-fd8ac840bc9e
# ╠═f2155b91-85bb-4da6-894c-1dc6a42f33c9
# ╠═24a169a1-9f40-47d6-811a-cc85248c71e9
# ╠═c6cccee4-7a36-4fee-b619-92550f9bd16a
# ╠═5e5488f8-12ba-41e2-b47c-c2b3e8937a3f
# ╠═f0dd4069-0fb0-43cb-b144-afa5aea029d1
# ╠═348a8746-6a42-4820-8f2e-f1b16e005537
# ╠═175f41ae-2fe3-4b75-ae19-b6384334e490
# ╠═5b8fd3fd-8f9f-431e-9aff-0df3f26c5f6e
# ╠═6d7efe46-4602-4282-97ef-4f0a56285589
# ╠═b60cd64b-0308-4040-a68f-6ca106a88dcb
# ╠═13c53788-b4d6-41ad-9a02-d76c1cf8473c
# ╠═eda46bb2-e07c-4ec8-8959-f5e52c8d5202
# ╠═2cf9a8ab-a709-430f-9564-8ca409e0c4c9
# ╠═54ce6edd-eae9-4bab-90f9-6e05c5647c98
# ╠═945ab911-b996-4354-ba8a-7987e9bdbd45
# ╟─c2efee16-04de-4c5d-848c-5e0346302cf9
# ╟─ff7551ce-531c-4322-8c9d-7204e8b7937d
# ╠═ec0c2989-c67d-4f33-85c9-5db53907a457
# ╠═e97336d9-f618-4c76-abac-6c934863834b
# ╠═019fe7bd-c001-4fa6-bef3-0f67cb49569c
# ╠═dc3e728f-4af4-42ec-9d67-0a09c125b826
# ╠═882edb35-5bd4-4248-bcfc-1b2dc899abb5
# ╠═e7d9860d-6072-4d14-838a-145d61bfac83
# ╠═d1c066fb-a42e-4010-9b4b-97c70f8e2931
# ╟─ba2d4dce-dae4-485f-b9ca-1cdc4c161cd3
# ╠═bbd6aa72-e1c1-4eb7-b887-8937f758af2b
# ╠═006fbbd2-fec1-489a-93c0-b047f7310588
# ╠═ab585b3f-5a93-4ae9-ad53-f1558654d6ef
# ╠═64705b6e-8de5-451f-99b2-f6e21c40b2f6
# ╠═5ad22545-a42e-4415-a22a-1821979f9645
# ╠═ff3f3a04-8240-4255-b7d7-d3ea91b035ed
# ╠═eb00ac7f-f436-476b-8a3c-ceb6f6ed7999
# ╠═5cf85642-e594-41ae-8e2c-e2403f67d128
# ╟─85fd1bdb-dd69-496e-905a-2c4359bfd6cc
# ╠═c957e3cc-bfc4-4d0c-b0cf-fad1dee38bba
# ╠═08e115c3-96a8-413a-85ec-6d62bfd4783a
# ╠═a870267e-21bb-43a2-95f9-9a027057c6c8
# ╠═bc3966fd-12e6-4b4a-85f7-0cfcd934244c
# ╠═cf39907e-83e2-4dfb-a4c4-cacbfdc4f75c
# ╟─8241bd0e-d5de-41a6-ad9c-d92c447e9f75
# ╠═87d62425-2d14-4643-b039-b2f1258a1b0a
# ╠═7f98c3fd-1841-4d39-8ad4-2446bee6285f
# ╠═61c1cb94-74ff-46e2-acc8-45ae27cdeb24
# ╠═4671521e-2df4-47d9-b2db-cf7f1fc3e86a
# ╟─2385f99a-b14b-41fb-9c83-d7463f1cf0d8
# ╟─e0b6bf9c-1e97-4ae0-9db1-7bd3bc574515
# ╟─c93de303-23c1-4485-b0be-c87b4ebaf6aa
# ╟─5ce219b6-850e-4a69-b78b-e17fc6c2b205
# ╟─08453927-b13b-4250-a25f-31911341f8bb
