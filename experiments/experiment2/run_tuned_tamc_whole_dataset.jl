include("../../src/QUBOPortfolio.jl")
include("../Dataset/load_dataset.jl")
include("../Plots/plot_execution_data.jl")
import JLD2

training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances()
tuned_heuristics = JLD2.load("cache/best_tamc_params_results_100s.jld2", "tuned_tamc_heuristics")
default_heuristic = QUBOPortfolio.load_default_tamc_heuristic()

all_heuristics = vcat(tuned_heuristics, [default_heuristic])
all_instances = vcat(training_instances, test_instances_mqlib, test_instances_qubolib)
all_instances = all_instances[1:600] # TODO decrease the limit for testing

cache = QUBOPortfolio.ResultCache()
training_results = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    all_instances;
    repeat=3,
    verbosity=true,
    cache=cache,
    max_concurrent_tasks=8 # 32 / 4 threads per heuristic = 8
)

print_performance_table(training_results)
# plot_execution_times(training_results, all_instances)
# plot_performance(training_results, all_instances; baseline_heuristic_name=default_heuristic.name, filter_heuristics=[])
# plot_histogram_execution_times(training_results, tuned_heuristics[1].name)
# plot_histogram_execution_times(training_results, tuned_heuristics[2].name)
# @info QUBOPortfolio.instances_with_time_above(training_results, 500.0)
