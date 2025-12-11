
function run_quboporfolio_on_instance(instance::QUBOPortfolio.Instance,
                                      heuristic::QUBOPortfolio.Heuristic,
                                      cache::Union{ResultCache, Nothing}=nothing)::Vector{QUBOPortfolio.Result}
    time_taken = @elapsed begin
        top_k = get(heuristic.hyperparameters, "top_k", 5)
        results = QUBOPortfolio.run_portfolio(heuristic.portfolio, [instance], top_k, cache)
        # only one instance was run
        execution_results = first(values(results))
        best_energy = Inf
        best_result = nothing
        all_results = []
        for (_, results_vec) in execution_results.results
            for res in results_vec
                push!(all_results, res)
            end
        end
    end

    # Adjust time taken to include portfolio selection time
    if cache !== nothing
        # Create a dummy result to store the total time taken including portfolio overhead
        used_heuristic = Heuristic(
            type=QUBOPortfolio.QUBOPORTFOLIO,
            name=heuristic.name,
            # clear portfolio as it is unused here
            portfolio=nothing,
            hyperparameters=heuristic.hyperparameters
        )
        # energy is the best energy found among all sub-heuristics
        energy = minimum([res.energy for res in all_results])
        push!(all_results, QUBOPortfolio.Result(energy=energy, time_taken=time_taken, used_heuristics=[used_heuristic]))
    end

    return all_results
end
