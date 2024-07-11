using Test

include("../testcases.jl")

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
    obs = get_observable(eks_sol, :obs)
    @test isa(obs, DimArray)
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
