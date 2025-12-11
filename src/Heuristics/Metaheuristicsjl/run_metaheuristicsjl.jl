

# accepts heuristics "PSO" and "BRKGA"
function run_PSO_on_instance(instance::QUBOPortfolio.Instance, hyperparameters::Dict{String, Any}, heuristic_name::String)::Result
    time_taken = @elapsed begin
        qubo_objective = prepare_qubo_objective(instance)
        bounds = prepare_bounds(instance)
        # common Metaheuristics.jl options from hyperparameters
        options = prepare_options(hyperparameters)

        # Set PSO-specific hyperparameters with defaults if not provided
        PSO_heuristic = Metaheuristics.PSO(
            N = get(hyperparameters, "N", 50),
            C1 = get(hyperparameters, "C1", 2.0),
            C2 = get(hyperparameters, "C2", 2.0),
            ω = get(hyperparameters, "ω", 0.8),
            options = options
        )

        result = Metaheuristics.optimize(qubo_objective, bounds, PSO_heuristic)
    end
    return QUBOPortfolio.Result(
        energy=minimum(result),
        solution=minimizer(result),
        time_taken=time_taken,
        used_heuristics=[Heuristic(type=QUBOPortfolio.METAHEURISTICSJL_PSO, name=heuristic_name, hyperparameters=hyperparameters)]
    )
end

function run_BRKGA_on_instance(instance::QUBOPortfolio.Instance, hyperparameters::Dict{String, Any}, heuristic_name::String)::Result
    time_taken = @elapsed begin
        qubo_objective = prepare_qubo_objective(instance)
        bounds = prepare_bounds(instance)
        # common Metaheuristics.jl options from hyperparameters
        options = prepare_options(hyperparameters)

        # Set BRKGA-specific hyperparameters with defaults if not provided
        BRKGA_heuristic = Metaheuristics.BRKGA(
            num_elites = get(hyperparameters, "num_elites", 20),
            num_mutants = get(hyperparameters, "num_mutants", 10),
            num_offsprings = get(hyperparameters, "num_offsprings", 70),
            bias = get(hyperparameters, "bias", 0.7),
            options = options
        )
        result = Metaheuristics.optimize(qubo_objective, bounds, BRKGA_heuristic)
    end
    return QUBOPortfolio.Result(
        energy=minimum(result),
        solution=minimizer(result),
        time_taken=time_taken,
        used_heuristics=[Heuristic(type=QUBOPortfolio.METAHEURISTICSJL_BRKGA, name=heuristic_name, hyperparameters=hyperparameters)]
    )
end


function prepare_qubo_objective(instance::QUBOPortfolio.Instance)
    linear_terms = Dict(QUBOTools.linear_terms(instance.qubo_instance))
    quadratic_terms = Dict(QUBOTools.quadratic_terms(instance.qubo_instance))

    all_vars = Set{Int}()
    for v in keys(linear_terms)
        push!(all_vars, v)
    end
    for (v1, v2) in keys(quadratic_terms)
        push!(all_vars, v1)
        push!(all_vars, v2)
    end

    variables = sort(collect(all_vars))

    var_map = Dict(v => i for (i, v) in enumerate(variables))

    function qubo_objective(x::AbstractVector)
        energy = 0.0
        # Add contributions from linear terms (h_i * x_i)
        for (var, coeff) in linear_terms
            idx = var_map[var]
            energy += coeff * x[idx]
        end
        # Add contributions from quadratic terms (J_ij * x_i * x_j)
        for ((var1, var2), coeff) in quadratic_terms
            idx1 = var_map[var1]
            idx2 = var_map[var2]
            energy += coeff * x[idx1] * x[idx2]
        end
        return energy
    end

    return qubo_objective
end

function prepare_bounds(instance::QUBOPortfolio.Instance)
    num_vars = QUBOPortfolio.num_qubo_variables(instance)
    bounds = hcat(zeros(Int, num_vars), ones(Int, num_vars))
    return bounds
end

function prepare_options(hyperparameters::Dict{String, Any})
    options = Metaheuristics.Options(
        f_calls_limit = get(hyperparameters, "f_calls_limit", 0),
        f_tol_rel = get(hyperparameters, "f_tol_rel", 1e-8),
        time_limit = get(hyperparameters, "time_limit", Inf),
        iterations = get(hyperparameters, "iterations", 2000),
        store_convergence = get(hyperparameters, "store_convergence", false),
        verbose = get(hyperparameters, "verbose", false),
    )
    return options
end
