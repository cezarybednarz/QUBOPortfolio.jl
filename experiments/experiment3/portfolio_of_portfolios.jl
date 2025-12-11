
include("../../src/QUBOPortfolio.jl")
include("../Portfolios/create_portfolios.jl")
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")
include("../Plots/plot_random_forest.jl")
import JLD2

# Heuristics
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(true)

tuned_sb_heuristics = JLD2.load("cache/best_sb_params_results.jld2", "tuned_sb_heuristics")
tuned_tamc_heuristics = JLD2.load("cache/best_tamc_params_results_100s.jld2", "tuned_tamc_heuristics")
default_sb_heuristic = QUBOPortfolio.load_default_sb_heuristic()
default_tamc_heuristic = QUBOPortfolio.load_default_tamc_heuristic()

all_mqlib_heuristics = QUBOPortfolio.load_8_best_mqlib_heuristics()
all_tamc_heuristics = vcat(tuned_tamc_heuristics, [default_tamc_heuristic])
all_sb_heuristics = vcat(tuned_sb_heuristics, [default_sb_heuristic])

cache = QUBOPortfolio.ResultCache()

# Run and save results for a given heuristic
function run_and_save(heuristic)
    results = QUBOPortfolio.run_heuristics_on_dataset(
        [heuristic],
        test_instances_mqlib;
        repeat=1,
        verbosity=true,
        cache=cache,
        readonly_cache=true,
    )
    filepath = "cache/ExecutionResults/$(heuristic.name)_test_results.jld2"
    QUBOPortfolio.save_results_to_file(filepath, results)
    @info "\nSaved results to $filepath\n"
    print_performance_table(results)
    return heuristic
end

#! All tested portfolio configurations

# Create MQLib based portfolio heuristic (Random Forest)
mqlib_portfolio_heuristic = run_and_save(create_and_train_portfolio_heuristic(
    all_mqlib_heuristics,
    training_instances,
    "MQLib_RandomForest_Portfolio";
    cache=cache,
    top_k=4,
    type=:random_forest,
    use_portfolio_cache=true
))

# Create TAMC based portfolio heuristic (Random Forest)
tamc_portfolio_heuristic = run_and_save(create_and_train_portfolio_heuristic(
    all_tamc_heuristics,
    training_instances,
    "TAMC_RandomForest_Portfolio";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
))

# Create SB based portfolio heuristic (Random Forest)
sb_portfolio_heuristic = run_and_save(create_and_train_portfolio_heuristic(
    all_sb_heuristics,
    training_instances,
    "SB_RandomForest_Portfolio";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
))

# Create MQLib based portfolio heuristic (XGBoost)
mqlib_portfolio_heuristic_xgb = run_and_save(create_and_train_portfolio_heuristic(
    all_mqlib_heuristics,
    training_instances,
    "MQLib_XGBoost_Portfolio";
    cache=cache,
    top_k=4,
    type=:xgboost,
    use_portfolio_cache=false
))

# Create TAMC based portfolio heuristic (XGBoost)
tamc_portfolio_heuristic_xgb = run_and_save(create_and_train_portfolio_heuristic(
    all_tamc_heuristics,
    training_instances,
    "TAMC_XGBoost_Portfolio";
    cache=cache,
    top_k=1,
    type=:xgboost,
    use_portfolio_cache=false
))

# Create SB based portfolio heuristic (XGBoost)
sb_portfolio_heuristic_xgb = run_and_save(create_and_train_portfolio_heuristic(
    all_sb_heuristics,
    training_instances,
    "SB_XGBoost_Portfolio";
    cache=cache,
    top_k=1,
    type=:xgboost,
    use_portfolio_cache=false
))

# Create Portfolio of Portfolios heuristic (TAMC + MQLib) (Random Forest)
portfolio_of_portfolios_cpu_heuristic = run_and_save(create_and_train_portfolio_heuristic(
    [mqlib_portfolio_heuristic, tamc_portfolio_heuristic],
    training_instances,
    "Portfolio_of_Portfolios_CPU_RandomForest";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
))

# Create Portfolio of Portfolios heuristic (TAMC + MQLib) (XGBoost)
portfolio_of_portfolios_cpu_heuristic_xgb = run_and_save(create_and_train_portfolio_heuristic(
    [mqlib_portfolio_heuristic, tamc_portfolio_heuristic],
    training_instances,
    "Portfolio_of_Portfolios_CPU_XGBoost";
    cache=cache,
    top_k=1,
    type=:xgboost,
    use_portfolio_cache=true
))


# Create Portfolio of Portfolios heuristic (TAMC + MQLib + SB) (Random Forest)
portfolio_of_portfolios_all_heuristic = run_and_save(create_and_train_portfolio_heuristic(
    [mqlib_portfolio_heuristic, tamc_portfolio_heuristic, sb_portfolio_heuristic],
    training_instances,
    "Portfolio_of_Portfolios_All_RandomForest";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
))

# Create Portfolio of All Heuristics (TAMC + MQLib + SB) (Random Forest)
portfolio_all_heuristic = run_and_save(create_and_train_portfolio_heuristic(
    vcat(all_mqlib_heuristics, all_tamc_heuristics, all_sb_heuristics),
    training_instances,
    "Portfolio_of_All_Heuristics_RandomForest";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
))

# Create Portfolio of All Heuristics (TAMC + MQLib + SB) (XGBoost)
portfolio_all_heuristic_xgb = run_and_save(create_and_train_portfolio_heuristic(
    vcat(all_mqlib_heuristics, all_tamc_heuristics, all_sb_heuristics),
    training_instances,
    "Portfolio_of_All_Heuristics_XGBoost";
    cache=cache,
    top_k=1,
    type=:xgboost,
    use_portfolio_cache=false
))


# get all heuristics above:
all_portfolio_heuristics = [
    mqlib_portfolio_heuristic,
    tamc_portfolio_heuristic,
    sb_portfolio_heuristic,
    mqlib_portfolio_heuristic_xgb,
    tamc_portfolio_heuristic_xgb,
    sb_portfolio_heuristic_xgb,
    portfolio_of_portfolios_cpu_heuristic,
    portfolio_of_portfolios_cpu_heuristic_xgb,
    portfolio_of_portfolios_all_heuristic,
    portfolio_all_heuristic,
    portfolio_all_heuristic_xgb
]

#! MQLib

plot_random_forest_tree(
    mqlib_portfolio_heuristic,
    all_mqlib_heuristics[1].name,
    1,
    1
)

plot_histogram_number_of_heuristics(test_results, "Portfolio_of_Portfolios_All_RandomForest"; skip_portfolio=false)
