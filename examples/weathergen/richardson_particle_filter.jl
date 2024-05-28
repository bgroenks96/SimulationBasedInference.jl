using DataFrames, CSV

include("richardson_precip.jl")

function richardson_particle_filter(precip_data::DataFrame; num_particles::Int=100, num_samples=100)
    t0 = precip_data.time[1]
    pr0 = precip_data.prec[1]
    init_obs = richardson_precip_obs(1, pr0)
    initialstate = PrecipState(pr0, t0)
    pf_state = Gen.initialize_particle_filter(richardson_model, (initialstate, 0), init_obs, num_particles)

    N = length(precip_data.time)-1
    for t in 1:N
        log_total_weight, log_normalized_weights = Gen.normalize_weights(pf_state.log_weights)
        @info "Starting particle filter step $t/$N ($(precip_data.time[t])); ESS=$(Gen.effective_sample_size(log_normalized_weights))"
        maybe_resample!(pf_state, ess_threshold=num_particles/2)
        obs_next = richardson_precip_obs(t, precip_data.prec[t+1])
        Gen.particle_filter_step!(pf_state, (initialstate, t,), (NoChange(), UnknownChange(),), obs_next)
    end

    return (Gen.sample_unweighted_traces(pf_state, num_samples), initialstate, pf_state)
end

precip_data_pluvio = DataFrame(CSV.File("examples/data/ddm/NYA_pluvio_l1_precip_daily_v00_2017-2022.csv"))
precip_data_pluvio_2018 = filter(row -> year(row.time) == 2018, precip_data_pluvio)

Plots.plot(precip_data_pluvio_2018.prec)

# Doesn't seem to be working for time scales beyond 1-2 months; classic particle filter degeneracy problem, I think.
samples, initialstate, pf_state = richardson_particle_filter(precip_data_pluvio_2018, num_particles=1000);
para_ens = map(x -> get_para_from(x), samples)
new_trace, _ = generate(richardson_model, (initialstate, 29), para_ens[1])
Plots.plot(get_precip_from(new_trace))
