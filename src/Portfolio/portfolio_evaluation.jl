
function get_prob(prob)
    if true in MLJ.levels(prob)
        return MLJ.pdf(prob, true)
    else
        return 0.0
    end
end

# Return top_k results for top_k heuristics predicted by the portfolio
function run_portfolio(algorithm_portfolio::AlgorithmPortfolio, instances::Vector, top_k=3, cache::Union{ResultCache, Nothing}=nothing)
    # classify the instances
    metrics_df = classify_to_df(instances, algorithm_portfolio.cached_metrics)

    # get prediction whether the heuristic is the best for each instance
    instance_heuristics_match = [[] for _ in 1:length(instances)]
    for heuristic in algorithm_portfolio.heuristics
        model = algorithm_portfolio.models[heuristic.name]
        predictions = MLJ.predict(model, metrics_df)

        for (i, prediction) in enumerate(predictions)
            push!(instance_heuristics_match[i], prediction)
        end
    end

    # calculate the top_k performing heuristic for each instance
    results_for_instance = Dict{String, Any}()

    for instance_idx in 1:length(instances)
        instance_name = get_instance_name(instances[instance_idx])
        heuristics_match = get_prob.(instance_heuristics_match[instance_idx])
        top_k_heuristics_idxs = sortperm(heuristics_match, rev=true)[1:top_k]
        top_k_heuristics = [
            algorithm_portfolio.heuristics[i] for i in top_k_heuristics_idxs
        ]

        @info "Portfolio with heuristics: $(join([h.name for h in algorithm_portfolio.heuristics], ", ")), Top $top_k heuristics for $instance_name: $(join([h.name for h in top_k_heuristics], ", "))"
        # run those heuristics on the instance and get the best result
        results = run_heuristics_on_dataset(
            top_k_heuristics, [instances[instance_idx]]; repeat=1, verbosity=false,
            cache=cache,
            readonly_cache=true
        )
        results_for_instance[instance_name] = results
    end

    return results_for_instance
end

function create_selector_training_data(heuristics::Vector{Heuristic},
                                       instance_metrics::Dict{String, Vector{Metric}},
                                       results::ExecutionResults;
                                       target_function::TargetFunction=QUBOPortfolio.HIGHEST_MEAN,
                                       repeat::Int=5)
    # get metric names
    metric_names = map(metric -> metric.name, first(values(instance_metrics)))
    # prepare empty dataframes for each heuristic
    training_data = Dict{String, DataFrame}()
    for heuristic in heuristics
        training_data[heuristic.name] = DataFrame()
        training_data[heuristic.name][!, target_function.name] = Bool[]
        for metric_name in metric_names
            training_data[heuristic.name][!, metric_name] = Float64[]
        end
    end

    # calculate target results for each (heuristic, instance) pair.
    # For example, if a target_function is highest_mean_of_5, then for each (heuristic, instance)
    # we calculate the mean of the 5 results from the execution_results
    execution_target_results = Dict{Tuple{String, String}, Float64}()
    for (key, res) in results.results
        @info "Calculating target function for heuristic $(key.heuristic_name) on instance $(key.instance_name)"
        for r in res
            @info "  Energy: $(r.energy), Time: $(r.time_taken), Used Heuristics: $([h.name for h in r.used_heuristics])"
        end
        # filter only results produced by the given heuristic (the other resultsa are from the portfolio runs)
        energies = [r.energy for r in res if r.used_heuristics == [key.heuristic_name]]
        execution_target_results[(key.heuristic_name, key.instance_name)] = target_function.func(res)
        @info "   -> Target result: $(execution_target_results[(key.heuristic_name, key.instance_name)])"
    end

    # calculate highest mean-of-5 (or mean-of-$repeat_rate) for each instance
    for (instance_name, metrics) in instance_metrics
        sorted_results = sort(collect(execution_target_results[(heuristic.name, instance_name)] for heuristic in heuristics))
        best_heuristics = [heuristic for heuristic in heuristics if execution_target_results[(heuristic.name, instance_name)] == sorted_results[1]]
        @info "Best heuristics for instance $instance_name: $(join([h.name for h in best_heuristics], ", "))"
        for heuristic in heuristics
            row_data = Dict{Symbol, Any}(target_function.name => (heuristic in best_heuristics))
            for (i, metric_name) in enumerate(metric_names)
                row_data[Symbol(metric_name)] = metrics[i].value
            end
            push!(training_data[heuristic.name], row_data)
        end
    end

    return training_data
end
