function load_all_mqlib_heuristics()::Vector{QUBOPortfolio.Heuristic}
    heuristics = MQLib.heuristics()
    return [QUBOPortfolio.Heuristic(type=QUBOPortfolio.MQLIB, name=heuristic) for heuristic in heuristics]
end

function load_8_best_mqlib_heuristics()::Vector{QUBOPortfolio.Heuristic}
    all_mqlib_heuristics = load_all_mqlib_heuristics()
    selected_heuristics = Vector{QUBOPortfolio.Heuristic}()
    for heuristic in all_mqlib_heuristics
        # Best heuristics according to MQLib paper (from best to worst):
        if heuristic.name in ["BURER2002", "FESTA2002VNSPR", "PALUBECKIS2004bMST3", "FESTA2002GPR", "FESTA2002GVNS", "PALUBECKIS2004bMST2", "BEASLEY1998TS", "LU2010"]
            push!(selected_heuristics, heuristic)
        end
    end
    return selected_heuristics
end

function load_default_tamc_heuristic()::QUBOPortfolio.Heuristic
#     PT:
#   num_sweeps: 2000
#   warmup_fraction: 0.5
#   beta:
#     Geometric:
#       beta_min: 0.2
#       beta_max: 5.0
#       num_beta: 32
#   lo_beta: 1.0
#   icm: true
#   num_replica_chains: 2
#   threads: 4
#   sample: 32
#   sample_states: 32
#   sample_limiting: 2
    tamc_hyperparameters = Dict{String, Any}(
        "PT" => Dict{String, Any}(
            "num_sweeps" => 2000,
            "warmup_fraction" => 0.5,
            "lo_beta" => 1.0,
            "icm" => true,
            "threads" => 4, # This is the only hyperparameter we change from the original default
            "sample" => 32,
            "sample_states" => 32,
            "sample_limiting" => 2,
            "num_replica_chains" => 2,
            "beta" => Dict("Geometric" => Dict("beta_min" => 0.2, "beta_max" => 5.0, "num_beta" => 32))
        )
    )
    return QUBOPortfolio.Heuristic(type=QUBOPortfolio.TAMC, name="TAMC default", hyperparameters=tamc_hyperparameters)
end

function load_default_sb_heuristic()::QUBOPortfolio.Heuristic
    # The defaults are included in the `simulated_bifurcation` package itself
    return QUBOPortfolio.Heuristic(type=QUBOPortfolio.SIMULATED_BIFURCATION, name="SB default")
end

function load_all_tamc_heuristics()::Vector{QUBOPortfolio.Heuristic}
    # Common base settings for TAMC
    base_tamc_hyperparameters = Dict{String, Any}(
        "PT" => Dict{String, Any}(
            "num_sweeps" => 250,
            "warmup_fraction" => 0.5,
            "lo_beta" => 1.0,
            "icm" => true,
            "threads" => 4,
            "sample" => 32,
            "sample_states" => 32,
            "sample_limiting" => 2,
            "num_replica_chains" => 512,
            "beta" => Dict("Geometric" => Dict("beta_min" => 0.2, "beta_max" => 5.0, "num_beta" => 32))
        )
    )

    # Set 1: No ICM
    tamc_no_icm = deepcopy(base_tamc_hyperparameters)
    tamc_no_icm["PT"]["icm"] = false

    # Set 2: Wide Beta range
    tamc_wide_beta = deepcopy(base_tamc_hyperparameters)
    tamc_wide_beta["PT"]["beta"] = Dict("Geometric" => Dict("beta_min" => 0.05, "beta_max" => 10.0, "num_beta" => 32))

    # Set 3: More replicas, fewer sweeps
    tamc_more_replicas_fewer_sweeps = deepcopy(base_tamc_hyperparameters)
    tamc_more_replicas_fewer_sweeps["PT"]["num_sweeps"] = 125
    tamc_more_replicas_fewer_sweeps["PT"]["num_replica_chains"] = 512

    # Set 4: More sweeps, less replicas
    tamc_more_sweeps_less_replicas = deepcopy(base_tamc_hyperparameters)
    tamc_more_sweeps_less_replicas["PT"]["num_sweeps"] = 500
    tamc_more_sweeps_less_replicas["PT"]["num_replica_chains"] = 128

    return [QUBOPortfolio.Heuristic(type=QUBOPortfolio.TAMC, name="TAMC no ICM", hyperparameters=tamc_no_icm),
            QUBOPortfolio.Heuristic(type=QUBOPortfolio.TAMC, name="TAMC wide beta", hyperparameters=tamc_wide_beta),
            QUBOPortfolio.Heuristic(type=QUBOPortfolio.TAMC, name="TAMC more replicas, fewer sweeps", hyperparameters=tamc_more_replicas_fewer_sweeps),
            QUBOPortfolio.Heuristic(type=QUBOPortfolio.TAMC, name="TAMC more sweeps, less replicas", hyperparameters=tamc_more_sweeps_less_replicas)]
end

function load_all_metaheuristicsjl_heuristics()::Vector{QUBOPortfolio.Heuristic}
    return [
        QUBOPortfolio.Heuristic(type=QUBOPortfolio.METAHEURISTICSJL_BRKGA, name="MetaheuristicsJL BRKGA", hyperparameters=Dict()),
        QUBOPortfolio.Heuristic(type=QUBOPortfolio.METAHEURISTICSJL_PSO, name="MetaheuristicsJL PSO", hyperparameters=Dict())
    ]
end

function load_all_simulated_bifurcation_heuristics()::Vector{QUBOPortfolio.Heuristic}
    sb_ballistic_hyperparameters = Dict{String, Any}(
        "mode" => "discrete",
        "heated" => false,
        "time_step" => 0.1,
        "pressure_slope" => 0.01,
        "heat_coefficient" => 0.06,
        "max_steps" => 500
    )
    sb_discrete_hyperparameters = Dict{String, Any}(
        "mode" => "discrete",
        "heated" => false,
        "time_step" => 0.1,
        "pressure_slope" => 0.01,
        "heat_coefficient" => 0.06,
        "max_steps" => 500
    )
    return [
        QUBOPortfolio.Heuristic(type=QUBOPortfolio.SIMULATED_BIFURCATION, name="SB Ballistic", hyperparameters=sb_ballistic_hyperparameters),
        QUBOPortfolio.Heuristic(type=QUBOPortfolio.SIMULATED_BIFURCATION, name="SB Discrete", hyperparameters=sb_discrete_hyperparameters)
    ]
end
