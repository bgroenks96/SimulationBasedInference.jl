# # [Getting started: Linear ODE inversion](@id linearode)
# In this example, we will use `SimulationBasedInference` in combination with
# the `OrdinaryDiffEq` package to recover the true parameter of a simple linear
# ordinary differential equation:
# ```math
# \frac{\partial u}{\partial t} = -\alpha u
# ```
# Of course, this ODE has an analytical solution: $u(t) = u_0 e^{-\alpha t}$
# which we could (more efficiently) use to define the inverse problem. However,
# in order to demonstrate the usage of `SimulationBasedInference` on dynamical
# systems more broadly, we will solve the problem using numerical methods.

import Pkg; Pkg.instantiate() #hide

# First, we load the necessary packages
using SimulationBasedInference
using OrdinaryDiffEq
using Plots, StatsPlots
import Random

# extensions
using DynamicHMC

using DisplayAs #hide

# and then initialize a random number generator for reproducibility.
const rng = Random.MersenneTwister(1234);

# Now, we will define our simple dynamical system using the general SciML problem interface:
ode_func(u,p,t) = -p[1]*u;
α_true = 0.2
ode_p = [α_true];
tspan = (0.0,10.0);
odeprob = ODEProblem(ode_func, [1.0], tspan, ode_p)

# To solve an inverse (or inference) problem with `SimulationBasedInference`, we must first define
# a forward problem. The forward problem consists of two components:
# 1. A SciML problem type or forward map function $f: \Theta \mapsto \mathcal{U}$.
# 2. One or more "observables" which define the observation operator that transforms the model state to
# to something comparable to data.
#
# In this case, we can simply define a function that extracts the current
# state from the ODE integrator. The `SimulatorObservable(name, func, t0, tsave, coords)` additionally takes
# an initial time point, a vector of observed time points, and a tuple specifying the shape or coordiantes of
# the observable at each time point. Here, `(1,)` indicates that the state is a one-dimensional vector.
dt = 0.2
tsave = tspan[begin]+dt:dt:tspan[end];
n_obs = length(tsave);
observable = ODEObservable(:y, odeprob, tsave, samplerate=0.01);
forward_prob = SimulatorForwardProblem(odeprob, observable)

# In order to set up our synthetic example, we need some data to condition on.
# For this example, we will generate the data by running the forward model and adding Gaussian noise.
ode_solver = Tsit5()
forward_sol = solve(forward_prob, ode_solver);
true_obs = get_observable(forward_sol, :y)
noise_scale = 0.05
noisy_obs = true_obs .+ noise_scale*randn(rng, n_obs);
# Plot the resulting pseudo-observations vs. the ground truth.
plot(true_obs, label="True solution", linewidth=3, color=:black)
plt = scatter!(tsave, noisy_obs, label="Noisy observations", alpha=0.5)
DisplayAs.Text(DisplayAs.PNG(plt)) #hide

# Here we set our priors. We use a weakly informative `Beta(2,2)` prior which places
# less probability mass at the extreme values near zero and one. We could also use a flat
# prior `Beta(1,1)` if we wanted to be maximally indifferent to which parameters are most likely.
model_prior = prior(α=Beta(2,2));
noise_scale_prior = prior(σ=Exponential(0.1));
p1 = Plots.plot(model_prior.dist.α)
p2 = Plots.plot(noise_scale_prior.dist.σ)
plt = Plots.plot(p1, p2)
DisplayAs.Text(DisplayAs.PNG(plt)) #hide

# Now we assign a simple Gaussian likelihood for the observation/noise model.
lik = IsotropicGaussianLikelihood(observable, noisy_obs, noise_scale_prior);
nothing #hide

# We now have all of the ingredients needed to set up and solve the inference problem.
# We will start with a simple ensemble importance sampling inference algorithm.
inference_prob = SimulatorInferenceProblem(forward_prob, ode_solver, model_prior, lik);
enis_sol = solve(inference_prob, EnIS(), ensemble_size=128, rng=rng);
nothing #hide

# We can extract the prior ensemble from the solution.
prior_ens = get_transformed_ensemble(enis_sol)
prior_ens_mean = mean(prior_ens, dims=2)[:,1]
prior_ens_std = std(prior_ens, dims=2)[:,1]
prior_ens_obs = Array(get_observables(enis_sol).y);
prior_ens_obs_mean = mean(prior_ens_obs, dims=2)[:,1]
prior_ens_obs_std = std(prior_ens_obs, dims=2)
nothing #hide

# In contrast to other inference algorithms, importance sampling produces weights rather than
# direct samples from the posterior. We can use the `get_weights` method to extract these from the solution.
importance_weights = get_weights(enis_sol);
nothing #hide

# We can then use these importance weights to compute weighted statistics using standard methods from the `Statistics`
# and `StatsBase` modules. Note that these modules are exported by `SimulationBasedInference` for convenience.
posterior_obs_mean_enis = mean(prior_ens_obs, weights(importance_weights), 2)[:,1]
posterior_obs_std_enis = std(prior_ens_obs, weights(importance_weights), 2)[:,1]
posterior_mean_enis = mean(prior_ens, weights(importance_weights))

# Now we plot the prior vs. the posterior predictions.
plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="Linear ODE posterior predictions (EnIS)")
plot!(tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.5, linewidth=2)
plot!(tsave, posterior_obs_mean_enis, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_enis, alpha=0.7, linewidth=2)
plt = scatter!(tsave, noisy_obs, label="Noisy observations", c=:orange)
DisplayAs.Text(DisplayAs.PNG(plt)) #hide

# One of the key benefits of the standardized problem type interface is that we can very easily switch to
# a different algorithm by changing a single line of code. Here, we solve the same inference problem
# instead with the ensemble smoother w/ "multiple data assimilation" (ES-MDA).
esmda_sol = @time solve(inference_prob, ESMDA(), ensemble_size=128, rng=rng);
nothing #hide

# Now we extract the posterior ensemble and compute the relevant statistics.
posterior_esmda = get_transformed_ensemble(esmda_sol)
posterior_mean_esmda = mean(posterior_esmda, dims=2)

# Plotting the predictions shows that we get a much tighter estimate of the posterior mean.
posterior_obs_esmda = get_observables(esmda_sol).y
posterior_obs_mean_esmda = mean(posterior_obs_esmda, dims=2)[:,1]
posterior_obs_std_esmda = std(posterior_obs_esmda, dims=2)[:,1]
plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="ES-MDA")
plot!(tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.5, linewidth=2)
plot!(tsave, posterior_obs_mean_esmda, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_esmda, alpha=0.7, linewidth=2)
plt = scatter!(tsave, noisy_obs, label="Noisy observations", c=:black)
DisplayAs.Text(DisplayAs.PNG(plt)) #hide

# We can again solve the same problem with the Ensemble Kalman Sampler of Garbuno-Inigo et al. (2020)
# which yields very similar results (in this case).
eks_sol = @time solve(inference_prob, EKS(), ensemble_size=128, rng=rng, verbose=false)
posterior_eks = get_transformed_ensemble(eks_sol)
posterior_mean_eks = mean(posterior_eks, dims=2)
posterior_obs_eks = get_observables(eks_sol).y
posterior_obs_mean_eks = mean(posterior_obs_eks, dims=2)[:,1]
posterior_obs_std_eks = std(posterior_obs_eks, dims=2)[:,1]
plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="EKS")
plot!(tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.5, linewidth=2)
plot!(tsave, posterior_obs_mean_eks, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_eks, alpha=0.7, linewidth=2)
plt = scatter!(tsave, noisy_obs, label="Noisy observations", c=:black)
DisplayAs.Text(DisplayAs.PNG(plt)) #hide

# Now solve using the gold standard No U-turn sampler (NUTS). This will take a few minutes to run.
# Note that this would generally not be feasible for more expensive simulators.
hmc_sol = @time solve(inference_prob, MCMC(NUTS()), num_samples=1000, rng=rng);
posterior_hmc = transpose(Array(hmc_sol.result))
posterior_mean_hmc = mean(posterior_hmc, dims=2)
posterior_obs_hmc = reduce(hcat, map(out -> out.y, hmc_sol.storage.outputs))
posterior_obs_mean_hmc = mean(posterior_obs_hmc, dims=2)[:,1]
posterior_obs_std_hmc = std(posterior_obs_hmc, dims=2)[:,1]
plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="EKS")
plot!(tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.5, linewidth=2)
plot!(tsave, posterior_obs_mean_hmc, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_hmc, alpha=0.7, linewidth=2)
plt = scatter!(tsave, noisy_obs, label="Noisy observations", c=:black)

# Finally, we can plot the posterior predictions of all of the algorithms and compare.
plot(tsave, true_obs, label="True solution", c=:black, linewidth=2, title="Linear ODE: Inference algorithm comparison", dpi=300, xlabel="time")
plot!(tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.4, linewidth=2)
plot!(tsave, posterior_obs_mean_enis, label="Posterior (EnIS)", linestyle=:dash, ribbon=2*posterior_obs_std_enis, alpha=0.4, linewidth=3)
plot!(tsave, posterior_obs_mean_esmda, label="Posterior (ES-MDA)", linestyle=:dash, ribbon=2*posterior_obs_std_esmda, alpha=0.4, linewidth=3)
plot!(tsave, posterior_obs_mean_eks, label="Posterior (EKS)", linestyle=:dash, ribbon=2*posterior_obs_std_eks, alpha=0.4, linewidth=3)
plot!(tsave, posterior_obs_mean_hmc, label="Posterior (HMC)", linestyle=:dash, ribbon=2*posterior_obs_std_hmc, alpha=0.4, linewidth=3)
plt = scatter!(tsave, noisy_obs, label="Noisy observations", c=:black)
filepath = "res/linearode_poseterior_preds_comparison.png" #hide
mkpath(dirname(filepath)) #hide
savefig(filepath) #hide
DisplayAs.Text(DisplayAs.PNG(plt)) #hide
