using Hyperopt
using Random
# using DataStructures # For ConcurrentStack (if not exported by QUBOPortfolio) - Removed as requested

# The average energy over the instances (to be minimized)
# Implements the loss function: L(θ) = E(θ) + ρ * max(0, time - limit)
function average_energy_for_hyperparameters_on_instances(heuristic_type::QUBOPortfolio.HeuristicType,
                                                        instances,
                                                        hyperparameters,
                                                        time_limit::Float64=100.0)::Float64
    energies = QUBOPortfolio.run_heuristics_on_dataset(
        [QUBOPortfolio.Heuristic(type=heuristic_type, name="$(string(heuristic_type))", hyperparameters=hyperparameters)],
        instances;
        repeat=1,
        verbosity=false,
        cache=nothing,
        max_concurrent_tasks=4 # TODO: parametrize it (32 threads / 2 tuned heuristics / 4 threads per TAMC run)
    )
    max_time = QUBOPortfolio.max_time(energies)
    total_energy = QUBOPortfolio.total_energy(energies)
    average_energy = total_energy / length(instances)

    # Penalize longer runs (ρ = 100.0)
    if max_time > time_limit
        penalty = 100.0 * (max_time - time_limit)
        @warn "Hyperparameter set $(hyperparameters) took too long time: $(max_time) seconds (greater than $(time_limit) seconds)."
        @warn "Adding penalty $(penalty)."
        return average_energy + penalty
    end
    return average_energy
end

# Run hyperparameter optimization for TAMC heuristic using BOHB
function run_hyperparameter_optimization_for_tamc(instances::Vector{<:QUBOPortfolio.AbstractInstance}, number_of_iterations::Int, tamc_icm::Bool, time_limit::Float64=100.0)::Dict{String, Any}

    # Define categorical options
    replica_chain_options = [16, 32, 64, 128, 256]
    num_options_chains = length(replica_chain_options)

    # 7 dimensions in the search space below
    bohb_dims = [Hyperopt.Continuous() for _ in 1:7]

    # Use Hyperband with BOHB as the inner optimizer
    ho = @hyperopt for resources = number_of_iterations,
                    sampler = Hyperband(R=number_of_iterations, η=3, inner=BOHB(dims=bohb_dims)),
                    num_sweeps_raw = LinRange(25.0, 250.0, 1000),
                    warmup_fraction = LinRange(0.1, 0.9, 1000),
                    lo_beta = LinRange(0.5, 1.5, 1000),
                    num_replica_chains_idx = LinRange(1.0, Float64(num_options_chains), 1000), # Continuous index for categorical
                    beta_min = LinRange(0.1, 1.0, 1000),
                    beta_max = LinRange(2.0, 10.0, 1000),
                    num_beta_raw = LinRange(16.0, 64.0, 1000)

                # If BOHB has a suggestion (state), we use it to overwrite the loop variables
                if state !== nothing
                    num_sweeps_raw, warmup_fraction, lo_beta, num_replica_chains_idx, beta_min, beta_max, num_beta_raw = state
                end

                # Integer Parameters: Round nearest
                num_sweeps = round(Int, num_sweeps_raw)
                num_beta = round(Int, num_beta_raw)

                # Categorical Parameters: Round to index and select
                chain_idx = clamp(round(Int, num_replica_chains_idx), 1, num_options_chains)
                num_replica_chains = replica_chain_options[chain_idx]

                hyperparameters = Dict(
                    "PT" => Dict(
                        "num_sweeps" => num_sweeps,
                        "warmup_fraction" => warmup_fraction,
                        "lo_beta" => lo_beta,
                        "icm" => tamc_icm,
                        "threads" => 4,
                        "sample" => 32,
                        "sample_states" => 32,
                        "sample_limiting" => 2,
                        "num_replica_chains" => num_replica_chains,
                        "beta" => Dict("Geometric" => Dict("beta_min" => beta_min, "beta_max" => beta_max, "num_beta" => num_beta))
                    )
                )

                loss = average_energy_for_hyperparameters_on_instances(QUBOPortfolio.TAMC, instances, hyperparameters, time_limit)

                loss, [num_sweeps_raw, warmup_fraction, lo_beta, num_replica_chains_idx, beta_min, beta_max, num_beta_raw]
            end # End @hyperopt macro

    # Retrieve best parameters (these are the raw continuous values from the optimizer)
    best_params_raw = ho.minimizer

    # Re-convert best raw parameters to actual heuristic parameters
    b_sweeps_raw = best_params_raw[1]
    b_warmup = best_params_raw[2]
    b_lo_beta = best_params_raw[3]
    b_chains_idx_raw = best_params_raw[4]
    b_beta_min = best_params_raw[5]
    b_beta_max = best_params_raw[6]
    b_num_beta_raw = best_params_raw[7]

    best_hyperparameters = Dict(
        "PT" => Dict(
            "num_sweeps" => round(Int, b_sweeps_raw),
            "warmup_fraction" => b_warmup,
            "lo_beta" => b_lo_beta,
            "icm" => tamc_icm,
            "threads" => 4,
            "sample" => 32,
            "sample_states" => 32,
            "sample_limiting" => 2,
            "num_replica_chains" => replica_chain_options[clamp(round(Int, b_chains_idx_raw), 1, num_options_chains)],
            "beta" => Dict("Geometric" => Dict(
                "beta_min" => b_beta_min,
                "beta_max" => b_beta_max,
                "num_beta" => round(Int, b_num_beta_raw)
            ))
        )
    )

    @info "Best hyperparameters for this subset (TAMC): $best_hyperparameters"
    return best_hyperparameters
end

# Run hyperparameter optimization for SIMULATED_BIFURCATION heuristic using BOHB
function run_hyperparameter_optimization_for_sb(instances::Vector{<:QUBOPortfolio.AbstractInstance}, number_of_iterations::Int, sb_heated::Bool)::Dict{String, Any}

    # 5 dimensions in the search space below
    bohb_dims = [Hyperopt.Continuous() for _ in 1:5]

    # Use Hyperband with BOHB as the inner optimizer
    ho = @hyperopt for resources = number_of_iterations,
            sampler = Hyperband(R=number_of_iterations, η=3, inner=BOHB(dims=bohb_dims)),
            time_step = LinRange(0.1, 1.0, 1000),
            pressure_slope = LinRange(0.001, 0.1, 1000),
            heat_coefficient = LinRange(0.01, 0.2, 1000),
            agents_raw = LinRange(200.0, 2000.0, 1000),
            max_steps_raw = LinRange(100.0, 1000.0, 1000)

            if state !== nothing
                time_step, pressure_slope, heat_coefficient, agents_raw, max_steps_raw = state
            end

            agents = round(Int, agents_raw)
            max_steps = round(Int, max_steps_raw)

            hyperparameters = Dict(
                "mode" => "discrete",
                "heated" => sb_heated,
                "time_step" => time_step,
                "pressure_slope" => pressure_slope,
                "heat_coefficient" => heat_coefficient,
                "agents" => agents,
                "max_steps" => max_steps
            )

            loss = average_energy_for_hyperparameters_on_instances(QUBOPortfolio.SIMULATED_BIFURCATION, instances, hyperparameters)

            # CRITICAL: Must return tuple of (loss, candidate_vector) for BOHB/Hyperband
            loss, [time_step, pressure_slope, heat_coefficient, agents_raw, max_steps_raw]
            end


    best_params_raw = ho.minimizer

    # Re-convert best raw parameters
    best_hyperparameters = Dict{String, Any}()
    best_hyperparameters["mode"] = "discrete"
    best_hyperparameters["heated"] = sb_heated
    best_hyperparameters["time_step"] = best_params_raw[1]
    best_hyperparameters["pressure_slope"] = best_params_raw[2]
    best_hyperparameters["heat_coefficient"] = best_params_raw[3]
    best_hyperparameters["agents"] = round(Int, best_params_raw[4])
    best_hyperparameters["max_steps"] = round(Int, best_params_raw[5])

    @info "Best hyperparameters for this subset (SB): $best_hyperparameters"
    return best_hyperparameters
end


# Tune the hyperparameters for a given heuristic type on a set of instances
#
# Params:
#  - heuristic_type: The type of heuristic to tune
#  - instances: The instances to use for tuning
#  - hyperparameters: The initial hyperparameters to tune
#  - size_of_instance_subsets: The size of subsets of instances to use for tuning
#  - number_of_subsets: The number of subsets to generate, and thus the number of
#    hyperparameter sets to find, and the number of heuristics to create
#  - number_of_iterations: The number of hyperparameter sets to try for each subset
#
# Returns: A vector of heuristics with tuned hyperparameters
function create_set_of_tuned_heuristics_for_type(heuristic_type::QUBOPortfolio.HeuristicType,
                              training_instances::Vector{<:QUBOPortfolio.AbstractInstance};
                              number_of_tuned_heuristics::Int = 3,
                              size_of_instance_subsets::Int = 10,
                              number_of_iterations::Int = 10,
                              tamc_icm::Bool = true,
                              sb_heated::Bool = false,
                              time_limit::Float64 = 100.0
                              )::Vector{QUBOPortfolio.Heuristic}

    hyperparameter_sets = ConcurrentStack{Dict{String, Any}}()

    Threads.@threads for i in 1:number_of_tuned_heuristics
        # Get a random subset of training instances
        training_subset = training_instances[randperm(length(training_instances))[1:size_of_instance_subsets]]

        @info "Tuning hyperparameters on subset $(i)/$(number_of_tuned_heuristics) using BOHB..."

        best_hyperparameters = if heuristic_type == QUBOPortfolio.TAMC
            run_hyperparameter_optimization_for_tamc(training_subset, number_of_iterations, tamc_icm, time_limit)
        elseif heuristic_type == QUBOPortfolio.SIMULATED_BIFURCATION
            run_hyperparameter_optimization_for_sb(training_subset, number_of_iterations, sb_heated)
        else
            error("Hyperparameter optimization not implemented (or removed) for heuristic type $(heuristic_type)")
        end

        push!(hyperparameter_sets, best_hyperparameters)
    end

    # Create heuristics structs
    tuned_heuristics = QUBOPortfolio.Heuristic[]
    i = 1
    while (maybe_hyperparameters = maybepop!(hyperparameter_sets)) !== nothing
        hyperparameters = something(maybe_hyperparameters) # Get the value from Some(value)
        heuristic = QUBOPortfolio.Heuristic(
            type=heuristic_type,
            name="$(string(heuristic_type)) - tuned set $(i)",
            hyperparameters=hyperparameters
        )
        push!(tuned_heuristics, heuristic)
        i += 1
    end

    return tuned_heuristics
end
