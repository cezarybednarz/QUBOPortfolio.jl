
include("../../src/QUBOPortfolio.jl")
include("../Portfolios/create_portfolios.jl")
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")
include("../Plots/plot_random_forest.jl")
import JLD2

# Heuristics
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(false)

tuned_sb_heuristics = JLD2.load("cache/best_sb_params_results.jld2", "tuned_sb_heuristics")
tuned_tamc_heuristics = JLD2.load("cache/best_tamc_params_results_100s.jld2", "tuned_tamc_heuristics")
default_sb_heuristic = QUBOPortfolio.load_default_sb_heuristic()
default_tamc_heuristic = QUBOPortfolio.load_default_tamc_heuristic()
default_brkga_heuristic = QUBOPortfolio.Heuristic(
    type=QUBOPortfolio.METAHEURISTICSJL_BRKGA,
    name="MetaheuristicsJL - BRKGA - default",
    hyperparameters=Dict{String, Any}() # empty dict means default hyperparameters
)
default_pso_heuristic = QUBOPortfolio.Heuristic(
    type=QUBOPortfolio.METAHEURISTICSJL_PSO,
    name="MetaheuristicsJL - PSO - default",
    hyperparameters=Dict{String, Any}() # empty dict means default hyperparameters
)


all_mqlib_heuristics = QUBOPortfolio.load_8_best_mqlib_heuristics()
all_tamc_heuristics = vcat(tuned_tamc_heuristics, [default_tamc_heuristic])
all_sb_heuristics = vcat(tuned_sb_heuristics, [default_sb_heuristic])
all_metaheuristics_heuristics = [default_brkga_heuristic, default_pso_heuristic]


cache = QUBOPortfolio.ResultCache()


all_heuristics_results = QUBOPortfolio.run_heuristics_on_dataset(
    vcat(all_mqlib_heuristics, all_tamc_heuristics, all_sb_heuristics, all_metaheuristics_heuristics),
    test_instances_mqlib;
    repeat=1,
    verbosity=true,
    cache=cache,
    readonly_cache=true
)

# Load MQLib based portfolio heuristic (Random Forest)
mqlib_portfolio_heuristic_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/MQLib_RandomForest_Portfolio_test_results.jld2")

# Load TAMC based portfolio heuristic (Random Forest)
tamc_portfolio_heuristic_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/TAMC_RandomForest_Portfolio_test_results.jld2")

# Load SB based portfolio heuristic (Random Forest)
sb_portfolio_heuristic_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/SB_RandomForest_Portfolio_test_results.jld2")

# Load MQLib based portfolio heuristic (XGBoost)
mqlib_portfolio_heuristic_xgb_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/MQLib_XGBoost_Portfolio_test_results.jld2")

# Load TAMC based portfolio heuristic (XGBoost)
tamc_portfolio_heuristic_xgb_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/TAMC_XGBoost_Portfolio_test_results.jld2")

# Load SB based portfolio heuristic (XGBoost)
sb_portfolio_heuristic_xgb_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/SB_XGBoost_Portfolio_test_results.jld2")

# Load Portfolio of Portfolios heuristic (TAMC + MQLib) (Random Forest)
portfolio_of_portfolios_cpu_heuristic_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/Portfolio_of_Portfolios_CPU_RandomForest_test_results.jld2")

# Load Portfolio of Portfolios heuristic (TAMC + MQLib) (XGBoost)
portfolio_of_portfolios_cpu_heuristic_xgb_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/Portfolio_of_Portfolios_CPU_XGBoost_test_results.jld2")

# Load Portfolio of Portfolios heuristic (TAMC + MQLib + SB) (Random Forest)
portfolio_of_portfolios_all_heuristic_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/Portfolio_of_Portfolios_All_RandomForest_test_results.jld2")

# Load Portfolio of Portfolios heuristic (TAMC + MQLib + SB) (XGBoost)
portfolio_of_portfolios_all_heuristic_xgb_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/Portfolio_of_Portfolios_All_XGBoost_test_results.jld2")

# Load Portfolio of All Heuristics (TAMC + MQLib + SB) (Random Forest)
portfolio_all_heuristic_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/Portfolio_of_All_Heuristics_RandomForest_test_results.jld2")

# Load Portfolio of All Heuristics (TAMC + MQLib + SB) (XGBoost)
portfolio_all_heuristic_xgb_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/Portfolio_of_All_Heuristics_XGBoost_test_results.jld2")

# Join all the results into one single ExecutionResults
all_results = QUBOPortfolio.join_execution_results([
    mqlib_portfolio_heuristic_results,
    tamc_portfolio_heuristic_results,
    sb_portfolio_heuristic_results,
    mqlib_portfolio_heuristic_xgb_results,
    tamc_portfolio_heuristic_xgb_results,
    sb_portfolio_heuristic_xgb_results,
    portfolio_of_portfolios_cpu_heuristic_results,
    portfolio_of_portfolios_cpu_heuristic_xgb_results,
    portfolio_of_portfolios_all_heuristic_results,
    portfolio_of_portfolios_all_heuristic_xgb_results,
    portfolio_all_heuristic_results,
    portfolio_all_heuristic_xgb_results
])

# Portfolios

all_results = QUBOPortfolio.join_execution_results([all_results, all_heuristics_results])

include("../Plots/plot_execution_data.jl")
plot_histogram_number_of_heuristics(all_results, "Portfolio_of_Portfolios_CPU_XGBoost"; skip_portfolio=false, filename="plots/Portfolio_of_Portfolios_CPU_XGBoost_heuristic_usage_histogram.png")
plot_histogram_number_of_heuristics(all_results, "Portfolio_of_All_Heuristics_RandomForest"; skip_portfolio=false, filename="plots/Portfolio_of_All_Heuristics_RandomForest_heuristic_usage_histogram.png")




# Heuristics + Portfolios
print_performance_table(all_results)
