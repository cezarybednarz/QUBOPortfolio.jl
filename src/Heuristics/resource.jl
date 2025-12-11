
Base.@kwdef struct Resource
    CPU::Float64 = 1.0
    GPU::Float64 = 0.0
end

function get_heuristic_resources(heuristic::QUBOPortfolio.Heuristic)::Resource
    if heuristic.type == QUBOPortfolio.TAMC
        return Resource(CPU=heuristic.hyperparameters["PT"]["threads"], GPU=0)
    elseif heuristic.type == QUBOPortfolio.MQLIB
        return Resource(CPU=1, GPU=0)
    elseif heuristic.type == QUBOPortfolio.SIMULATED_BIFURCATION
        return Resource(CPU=1, GPU=1)
    elseif heuristic.type == QUBOPortfolio.QUBOPORTFOLIO
        # QUBOPORTFOLIO itself uses resources for selection
        return Resource(CPU=1, GPU=0)
    else
        # Default resource allocation: MQLib heuristics, MetaheuristicsJL heuristics
        return Resource(CPU=1, GPU=0)
    end
end

function get_results_resources(results::Vector{Result})::Resource
    total_resources = Resource(CPU=0, GPU=0)
    max_portfolio_time = 0.0
    for result in results
        if result.used_heuristics[1].type == QUBOPortfolio.QUBOPORTFOLIO
            # Portfolio heuristic: take the max time among portfolio selections
            # This is a simplification for the case when a portfolio of portfolios chooses
            # just one child portfolio to run.
            @info "Portfolio $(result.used_heuristics[1].name) result time taken: $(result.time_taken)"
            max_portfolio_time = max(max_portfolio_time, result.time_taken)
        elseif length(result.used_heuristics) === 0
            @error "Result does not have associated heuristic information to determine resource usage."
        else
            # there is only one heuristic used per result in this context
            used_heuristic = result.used_heuristics[1]
            heuristic_resources = get_heuristic_resources(used_heuristic)
            cpu_seconds = result.time_taken * heuristic_resources.CPU
            gpu_seconds = result.time_taken * heuristic_resources.GPU
            total_resources = Resource(
                CPU=total_resources.CPU + cpu_seconds,
                GPU=total_resources.GPU + gpu_seconds
            )
        end
    end
    return Resource(
        CPU=total_resources.CPU + max_portfolio_time,
        GPU=total_resources.GPU
    )
end

# Get the maximum time taken among portfolio and heuristic results (to estimate total execution time)
function get_results_max_time(results::Vector{Result})::Float64
    max_portfolio_time = 0.0
    max_heuristic_time = 0.0
    for result in results
        if result.used_heuristics[1].type == QUBOPortfolio.QUBOPORTFOLIO
            max_portfolio_time = max(max_portfolio_time, result.time_taken)
        else
            max_heuristic_time = max(max_heuristic_time, result.time_taken)
        end
    end
    return max_portfolio_time + max_heuristic_time
end
