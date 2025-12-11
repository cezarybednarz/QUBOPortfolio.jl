#if !isdefined(Main, :QUBOPortfolio)
include("../../src/QUBOPortfolio.jl")
#end
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")

# 1. Create NewResults struct matching the new definition
# 2. Save the modified results to a new JLD2 file
# 3 Change the Result to match NewResult in the main codebase
# 4 Reload
# 5 Load the NewExecutionResults struct from JLD2 file and save it as a ExecutionResults (with Result struct)

results = JLD2.load("cache/results.jld2", "cache")

tuned_heuristics = JLD2.load("cache/best_sb_params_results.jld2", "tuned_sb_heuristics")
default_sb_heuristic = QUBOPortfolio.load_default_sb_heuristic()
default_tamc_heuristic = QUBOPortfolio.load_default_tamc_heuristic()

all_heuristics = vcat(QUBOPortfolio.load_8_best_mqlib_heuristics(), tuned_heuristics, [default_sb_heuristic, default_tamc_heuristic])

new_results = Dict{QUBOPortfolio.CacheKey, Vector{QUBOPortfolio.Result}}()

for (key, result_list) in results.results
    for result in result_list
        # Modify each Result to include an empty used_heuristics field
        heuristic = QUBOPortfolio.Heuristic(type=QUBOPortfolio.BRUTEFORCE)  # Placeholder initialization
        for h in all_heuristics
            if h.name == key.heuristic_name
                heuristic = h
                break
            end
        end
        if heuristic.type == QUBOPortfolio.BRUTEFORCE
            @error "Heuristic $(key.heuristic_name) not found in loaded heuristics."
            continue
        end
        # Add heuristic to used_heuristics field
        new_result = QUBOPortfolio.Result(
            energy = result.energy,
            solution = result.solution,
            time_taken = result.time_taken,
            used_heuristics = [heuristic]
        )
        result_list_for_key = get!(new_results, key, Vector{QUBOPortfolio.Result}())
        push!(result_list_for_key, new_result)
    end
end
cache = QUBOPortfolio.ResultCache()
full_execution_results = QUBOPortfolio.ExecutionResults(new_results)
QUBOPortfolio.save_results_to_cache(cache, full_execution_results)
QUBOPortfolio.load_results_from_cache(cache)
