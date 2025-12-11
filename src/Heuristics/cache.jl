# This file contains the caching mechanism for storing and retrieving heuristic execution
# results to file in order to avoid redundant computations.
import Base: hash, isequal

Base.@kwdef struct CacheKey
    type::HeuristicType
    heuristic_name::String
    hyperparameters::Dict{String, Any}
    instance_name::String
end

function hash(key::CacheKey, h::UInt)
    return hash(key.type, hash(key.heuristic_name, hash(key.hyperparameters, hash(key.instance_name, h))))
end

function isequal(a::CacheKey, b::CacheKey)
    return a.type == b.type &&
           a.heuristic_name == b.heuristic_name &&
           a.hyperparameters == b.hyperparameters &&
           a.instance_name == b.instance_name
end

struct ExecutionResults
    results::Dict{CacheKey, Vector{Result}}
end
ExecutionResults() = ExecutionResults(Dict{CacheKey, Vector{Result}}())

struct SharedExecutionResults
    results::ConcurrentDict{CacheKey, Vector{Result}}
end
SharedExecutionResults() = SharedExecutionResults(ConcurrentDict{CacheKey, Vector{Result}}())

function total_time_taken(results::ExecutionResults)::Float64
    total_time = 0.0
    for result_list in values(results.results)
        for result in result_list
            total_time += result.time_taken
        end
    end
    return total_time
end

function max_time(results::ExecutionResults)::Float64
    max_time = 0.0
    for result_list in values(results.results)
        for result in result_list
            if result.time_taken > max_time
                max_time = result.time_taken
            end
        end
    end
    return max_time
end

function total_energy(results::ExecutionResults)::Float64
    total_energy = 0.0
    for result_list in values(results.results)
        for result in result_list
            total_energy += result.energy
        end
    end
    return total_energy
end

function instances_with_time_above(results::ExecutionResults, threshold::Float64)::Set{String}
    instance_names = Set{String}()
    for (key, result_list) in results.results
        for result in result_list
            if result.time_taken > threshold
                push!(instance_names, key.instance_name)
                break
            end
        end
    end
    return instance_names
end


Base.@kwdef struct ResultCache
    filepath::String = "cache/results.jld2"
    snapshot_filepath::String = "cache/snapshot_results.jld2"
    lock::ReentrantLock = ReentrantLock()
end

function load_results_from_cache(cache::ResultCache)::ExecutionResults
    # CRITICAL FIX: Lock before checking file or loading
    lock(cache.lock) do
        if isfile(cache.filepath)
            try
                return JLD2.load(cache.filepath, "cache")
            catch e
                # If load fails (e.g. corrupted file from previous crash), return empty
                @warn "Failed to load cache from $(cache.filepath): $e. Creating a new cache."
                return ExecutionResults(Dict())
            end
        else
            return ExecutionResults(Dict())
        end
    end
end

function save_results_to_cache(cache::ResultCache, results::ExecutionResults)::Nothing
    # Using 'do' block automatically handles unlock, even on error
    lock(cache.lock) do
        try
            JLD2.save(cache.filepath, "cache", results)
        catch e
            @error "Failed to save cache to $(cache.filepath): $e"
        end
    end
    return nothing
end

function save_results_to_file(filepath::String, results::ExecutionResults)::Nothing
    try
        JLD2.save(filepath, "cache", results)
    catch e
        @error "Failed to save results to $(filepath): $e"
    end
    return nothing
end

function load_results_from_file(filepath::String)::ExecutionResults
    if isfile(filepath)
        try
            return JLD2.load(filepath, "cache")
        catch e
            @warn "Failed to load results from $(filepath): $e. Returning empty results."
            return ExecutionResults(Dict())
        end
    else
        return ExecutionResults(Dict())
    end
end

function convert_shared_to_results(shared_results::SharedExecutionResults)::ExecutionResults
    results = Dict{CacheKey, Vector{Result}}()
    for (key, value) in shared_results.results
        results[key] = value
    end
    return ExecutionResults(results)
end

function convert_results_to_shared(results::ExecutionResults)::SharedExecutionResults
    shared_results = ConcurrentDict{CacheKey, Vector{Result}}()
    for (key, value) in results.results
        shared_results[key] = value
    end
    return SharedExecutionResults(shared_results)
end

function join_execution_results(results_list::Vector{ExecutionResults})::ExecutionResults
    combined_results = Dict{CacheKey, Vector{Result}}()
    for results in results_list
        for (key, value) in results.results
            if haskey(combined_results, key)
                append!(combined_results[key], value)
            else
                combined_results[key] = copy(value)
            end
        end
    end
    return ExecutionResults(combined_results)
end
