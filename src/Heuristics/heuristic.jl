
# run a single heuristic on a single instance
# params:
# - heuristic: the heuristic to run
# - instance: the instance to run the heuristic on
# - binary_path: optional path to the binary to use for TAMC
# - cache: cache for QUBOPORTFOLIO heuristic type
function run_heuristic(heuristic::Heuristic,
                       instance::Instance,
                       binary_path::Union{String, Nothing}=nothing,
                       cache::Union{ResultCache, Nothing}=nothing
                       )::Vector{Result}
    time_taken = @elapsed begin
        results = begin
            if heuristic.type == QUBOPortfolio.TAMC
                if binary_path !== nothing
                    [QUBOPortfolio.run_tamc_on_instance(
                        instance,
                        heuristic.hyperparameters,
                        heuristic.name; tamc_path=binary_path
                    )]
                else
                    # Assuming run_tamc_on_instance can handle a default path if not provided
                    [QUBOPortfolio.run_tamc_on_instance(
                        instance,
                        heuristic.hyperparameters,
                        heuristic.name
                    )]
                end
            elseif heuristic.type == QUBOPortfolio.MQLIB
                [QUBOPortfolio.run_mqlib_on_instance(heuristic.name, instance)]
            elseif heuristic.type == QUBOPortfolio.METAHEURISTICSJL_PSO
                [QUBOPortfolio.run_PSO_on_instance(instance, heuristic.hyperparameters, heuristic.name)]
            elseif heuristic.type == QUBOPortfolio.METAHEURISTICSJL_BRKGA
                [QUBOPortfolio.run_BRKGA_on_instance(instance, heuristic.hyperparameters, heuristic.name)]
            elseif heuristic.type == QUBOPortfolio.BRUTEFORCE
                [QUBOPortfolio.run_bruteforce_on_instance(instance)]
            elseif heuristic.type == QUBOPortfolio.SIMULATED_BIFURCATION
                [QUBOPortfolio.run_simulated_bifurcation_on_instance(instance, heuristic.hyperparameters, heuristic.name)]
            elseif heuristic.type == QUBOPortfolio.QUBOPORTFOLIO
                # Returns vector of results from all ran heuristics in the portfolio
                QUBOPortfolio.run_quboporfolio_on_instance(instance, heuristic, cache)
            else
                error("Unknown heuristic type: $(heuristic.type)")
            end
        end
    end

    return results
end

function select_subset_of_results(results::ExecutionResults, heuristics, instances, repeat::Int)::ExecutionResults
    filtered_results = Dict{CacheKey, Vector{Result}}()
    for heuristic in heuristics
        for instance in instances
            instance_name = get_instance_name(instance)
            cache_key = CacheKey(
                type=heuristic.type,
                heuristic_name=heuristic.name,
                hyperparameters=heuristic.hyperparameters,
                instance_name=instance_name
            )
            if haskey(results.results, cache_key)
                if heuristic.type != QUBOPortfolio.QUBOPORTFOLIO
                    filtered_results[cache_key] = results.results[cache_key][1:repeat]
                else
                    # For QUBOPORTFOLIO, include all results (not just up to repeat)
                    filtered_results[cache_key] = results.results[cache_key]
                end
            end
        end
    end
    return ExecutionResults(filtered_results)
end

function run_heuristics_on_dataset(heuristics::Vector{QUBOPortfolio.Heuristic},
                    instances::Vector{<:QUBOPortfolio.AbstractInstance};
                    repeat::Int=1,
                    verbosity::Bool=true,
                    cache::Union{ResultCache, Nothing}=nothing,
                    max_concurrent_tasks::Int=Threads.nthreads(),
                    readonly_cache::Bool=false)::ExecutionResults
    if verbosity
        @info "Running $(length(heuristics)) heuristic(s) on $(length(instances)) instance(s). Each run will be repeated $repeat time(s)"
        @info "Using cache: $(cache !== nothing)"
        @info "Running with a maximum of $(max_concurrent_tasks) concurrent tasks. The available threads are: $(Threads.nthreads())"
        for heuristic in heuristics
            res = get_heuristic_resources(heuristic)
            @info "Heuristic $(heuristic.name) (type: $(heuristic.type)) uses resources: CPU=$(res.CPU), GPU=$(res.GPU)"
        end
    end

    results = ExecutionResults()

    # Load cached results from file if available
    if cache !== nothing
        try
            results = load_results_from_cache(cache)
            if verbosity
                @info "Loaded cached results and times from cache."
            end
        catch
            @warn "Failed to load cached results. Proceeding without cache."
        end
    end

    # Use ConcurrentDict for thread-safe writes
    shared_results = convert_results_to_shared(results)

    tasks = [(instance_id, heuristic) for instance_id in 1:length(instances) for heuristic in heuristics]

    finished_tasks = Threads.Atomic{Int}(0)
    total_tasks = length(tasks)

    task_channel = Channel{eltype(tasks)}(length(tasks))
    for task in tasks
        put!(task_channel, task)
    end
    close(task_channel)

    @sync for i in 1:max_concurrent_tasks
        Threads.@spawn for (instance_id, heuristic) in task_channel
            instance_name = get_instance_name(instances[instance_id])

            cache_key = CacheKey(
                type=heuristic.type,
                heuristic_name=heuristic.name,
                hyperparameters=heuristic.hyperparameters,
                instance_name=instance_name
            )

            cached_results = maybeget(shared_results.results, cache_key)
            cached_results = something(cached_results, Vector{Result}())

            # Do not load cached QUBOPORTFOLIO results.
            if heuristic.type == QUBOPortfolio.QUBOPORTFOLIO
                cached_results = Vector{Result}()
            # If cache contains enough results, skip.
            elseif heuristic.type !== QUBOPortfolio.QUBOPORTFOLIO && length(cached_results) >= repeat
                if verbosity
                    @info "Enough results for heuristic $(heuristic.name) on instance $(instance_name) found in cache. Skipping execution."
                end
                Threads.atomic_add!(finished_tasks, 1)
                continue
            end

            # Load instance data into memory before running heuristics.
            loaded_instance = get_instance_data(instances[instance_id])

            # Calculate the remaining runs needed
            remaining_runs = repeat - length(cached_results)
            if verbosity
                @info "Need to run $(heuristic.name) on instance $(instance_name) $(remaining_runs) time(s)."
            end
            for _ in 1:remaining_runs
                # Run heuristics
                new_results = run_heuristic(heuristic, loaded_instance, nothing, cache)
                cached_results = vcat(cached_results, new_results)
            end

            # Update shared results
            shared_results.results[cache_key] = cached_results

            # Save to cache every 5 iterations
            Threads.atomic_add!(finished_tasks, 1)
            task_number = finished_tasks[]
            if verbosity
                @info "\n=== Finished task $(task_number)/$(total_tasks) ($(round(task_number/total_tasks * 100, digits=2))%) ===\n"
            end
            if !readonly_cache && cache !== nothing && task_number % 5 == 0
                if verbosity
                    @info "Saving intermediate results to cache... (at task $(task_number)/$(total_tasks))"
                end
                save_results_to_cache(cache, convert_shared_to_results(shared_results))
            end
        end
    end

    # Final save to cache
    if !readonly_cache && cache !== nothing
        if verbosity
            @info "Saving final results to cache..."
        end
        save_results_to_cache(cache, convert_shared_to_results(shared_results))
    end
    all_results = convert_shared_to_results(shared_results)
    results = select_subset_of_results(all_results, heuristics, instances, repeat)
    return results
end
