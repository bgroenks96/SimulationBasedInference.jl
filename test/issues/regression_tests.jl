using Test

@testset "Issue #3" begin
    using SimulationBasedInference

    rng = Random.MersenneTwister(1234)
    # linear ODE test case with default parameter settings
    inference_prob = linear_ode(; rng)
    # parameter prior, excluding likelihood parameters
    prior = inference_prob.prior.model
    eks = EKS()
    # solve inference problem with EKS
    eks_sol = solve(inference_prob, eks, EnsembleThreads(), ensemble_size=128, verbose=false, rng=rng)
    eks_obs = get_observables(eks_sol)
    @test isa(eks_obs.obs, DimArray)
end

@testset "Issue #4" begin
    """
    Initial ensemble is not respected by ensemble solvers.
    """

    using SimulationBasedInference
    using OrdinaryDiffEq
    import Random
    
    rng = Random.MersenneTwister(1234);
    
    """Defining factory for the linear ode problem simulation. """
    function problem_factory(ode_func, t_data, solver)
    
        function problem_simulation(θ)
            prob = ODEProblem(ode_func, 1.0, (t_data[begin], t_data[end]), θ[1])
            sol = solve(prob, solver, saveat = t_data)
            return hcat(sol.u...)
        end
    
        return problem_simulation
    end
    
    # define ode_func
    ode_func(u,p,t) = -p[1]*u; 
    
    # Define true parameter
    α_true =[0.2]
    
    # Define time span and observation times
    tspan = (0.0,10.0)
    dt = 0.2
    tsave = tspan[begin]:dt:tspan[end]
    n_obs = length(tsave)
    
    # Define observable and forward problem
    observable = SimulatorObservable(:y, state -> state.u, (n_obs,))
    ode_solver = Tsit5()
    forward_prob = SimulatorForwardProblem(problem_factory(ode_func, tsave, ode_solver), α_true, observable)
    
    # Generating synthetic data by running the forward solution and adding noise
    forward_sol = solve(forward_prob);
    true_obs = get_observable(forward_sol, :y)
    noise_scale = 0.05
    noisy_obs = true_obs .+ noise_scale*randn(rng, n_obs);
    
    # Setting priors
    model_prior = prior(α=Beta(2,2));
    noise_scale_prior = prior(σ=Exponential(0.1));
    
    # Assign a simple Gaussian likelihood for the obsevation.
    lik = IsotropicGaussianLikelihood(observable, noisy_obs, noise_scale_prior);
    
    # We now have all of the ingredients needed to set up and solve the inference problem.
    # We will start with a simple ensemble importance sampling inference algorithm.
    inference_prob = SimulatorInferenceProblem(forward_prob, model_prior, lik)
    # sample initial ensemble
    ensemble_size = 8
    initial_ens = reduce(hcat, sample(rng, model_prior, ensemble_size))
    enis_sol = solve(inference_prob, EnIS(), rng=rng; initial_ens = initial_ens);
    true_ens = get_transformed_ensemble(enis_sol, 1)
    
    @test all(true_ens .≈ initial_ens)
end

@testset "Issue #12" begin
    using PythonCall

    using OrdinaryDiffEq

    using SimulationBasedInference
    using SimulationBasedInference.PySBI

    import Random

    rng = Random.MersenneTwister(1234);
    
    ode_func(u,p,t) = -p[1]*u;
    α_true = 0.2
    ode_p = [α_true];
    tspan = (0.0,10.0);
    odeprob = ODEProblem(ode_func, [1.0], tspan, ode_p)
    
    tsave = tspan[1]+0.1:0.2:tspan[end];
    n_obs = length(tsave);
    observable = SimulatorObservable(:y, integrator -> integrator.u, tspan[1], tsave, size(odeprob.u0), samplerate=0.01);
    
    forward_prob = SimulatorForwardProblem(odeprob, observable)
    ode_solver = Tsit5();
    forward_sol = solve(forward_prob, ode_solver);
    y_pred = get_observable(forward_sol, :y)
    
    true_obs = get_observable(forward_sol, :y)
    noisy_obs = true_obs .+ 0.05*randn(rng, n_obs);
    
    model_prior = prior(α=Beta(2,2));
    noise_scale_prior = prior(σ=Exponential(0.1));
    
    lik = IsotropicGaussianLikelihood(observable, noisy_obs, noise_scale_prior);
    
    inference_prob = SimulatorInferenceProblem(forward_prob, ode_solver, model_prior, lik);
    
    snpe_sol = solve(inference_prob, PySNE(), num_simulations=1000, rng=rng);    
end
