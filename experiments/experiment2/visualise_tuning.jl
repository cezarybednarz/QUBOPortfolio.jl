include("../../src/QUBOPortfolio.jl")
include("../Dataset/load_dataset.jl")
include("../Plots/plot_execution_data.jl")
import JLD2

# TAMC
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(false)
tuned_tamc_heuristics = JLD2.load("cache/best_tamc_params_results_100s.jld2", "tuned_tamc_heuristics")
default_tamc_heuristic = QUBOPortfolio.load_default_tamc_heuristic()

# SIMULATED_BIFURCATION
tuned_sb_heuristics = JLD2.load("cache/best_sb_params_results.jld2", "tuned_sb_heuristics")
default_sb_heuristic = QUBOPortfolio.load_default_sb_heuristic()

# Combine all heuristics
all_heuristics = vcat(
    tuned_tamc_heuristics,
    [default_tamc_heuristic],
    tuned_sb_heuristics,
    [default_sb_heuristic]
)

cache = QUBOPortfolio.ResultCache()
test_results = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    test_instances_mqlib;
    repeat=1,
    verbosity=true,
    cache=cache,
    readonly_cache=true,
    max_concurrent_tasks=6
)

train_results = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    training_instances;
    repeat=1,
    verbosity=true,
    cache=cache,
    readonly_cache=true,
    max_concurrent_tasks=6
)


@info "SB hyperparameters: $(tuned_sb_heuristics[1].hyperparameters) $(tuned_sb_heuristics[2].hyperparameters)"
@info "Default SB hyperparameters: $(default_sb_heuristic.hyperparameters)"

@info "TAMC hyperparameters: $(tuned_tamc_heuristics[1].hyperparameters) $(tuned_tamc_heuristics[2].hyperparameters)"
@info "Default TAMC hyperparameters: $(default_tamc_heuristic.hyperparameters)"


plot_execution_times(test_results, test_instances_mqlib)
plot_performance(test_results, test_instances_mqlib; baseline_heuristic_name=default_sb_heuristic.name, filter_heuristics=[])
plot_histogram_execution_times(test_results, default_tamc_heuristic.name; filename="plots/execution_time_histogram_tamc.png")
plot_histogram_execution_times(test_results, default_tamc_heuristic.name; filename="plots/execution_time_histogram_tamc.png")

plot_histogram_execution_times(train_results, default_sb_heuristic.name; filename="plots/training_execution_time_histogram_sb.png", distribution=true)
plot_histogram_execution_times(train_results, default_tamc_heuristic.name; filename="plots/training_execution_time_histogram_tamc.png", distribution=true)

plot_histogram_execution_times(test_results, tuned_sb_heuristics[1].name; filename="plots/execution_time_histogram_sb_tuned_1.png")
plot_histogram_execution_times(test_results, tuned_sb_heuristics[2].name; filename="plots/execution_time_histogram_sb_tuned_2.png")
plot_histogram_execution_times(test_results, tuned_tamc_heuristics[1].name; filename="plots/execution_time_histogram_tamc_tuned_1.png")
plot_histogram_execution_times(test_results, tuned_tamc_heuristics[2].name; filename="plots/execution_time_histogram_tamc_tuned_2.png")

print_performance_table(test_results;
    filter_heuristics=[default_tamc_heuristic.name, tuned_tamc_heuristics[1].name, tuned_tamc_heuristics[2].name]
)
print_performance_table(test_results;
    filter_heuristics=[default_sb_heuristic.name, tuned_sb_heuristics[1].name, tuned_sb_heuristics[2].name]
)
