
include("../../src/QUBOPortfolio.jl")
include("../Portfolios/create_portfolios.jl")
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")
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

tuned_tamc_heuristics[1].name


# Create MQLib based portfolio heuristic (Random Forest)
mqlib_portfolio_heuristic = create_and_train_portfolio_heuristic(
    all_mqlib_heuristics,
    training_instances,
    "MQLib_RandomForest_Portfolio";
    cache=cache,
    top_k=4,
    type=:random_forest,
    use_portfolio_cache=true
)

# Create TAMC based portfolio heuristic (Random Forest)
tamc_portfolio_heuristic = create_and_train_portfolio_heuristic(
    all_tamc_heuristics,
    training_instances,
    "TAMC_RandomForest_Portfolio";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
)

# Create SB based portfolio heuristic (Random Forest)
sb_portfolio_heuristic = create_and_train_portfolio_heuristic(
    all_sb_heuristics,
    training_instances,
    "SB_RandomForest_Portfolio";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
)

# Create Portfolio of Portfolios heuristic (TAMC + MQLib + SB) (Random Forest)
portfolio_of_portfolios_all_heuristic = create_and_train_portfolio_heuristic(
    [mqlib_portfolio_heuristic, tamc_portfolio_heuristic, sb_portfolio_heuristic],
    training_instances,
    "Portfolio_of_Portfolios_All_RandomForest";
    cache=cache,
    top_k=1,
    type=:random_forest,
    use_portfolio_cache=true
)


include("../Plots/plot_random_forest.jl")
plot_random_forest_tree(portfolio_of_portfolios_all_heuristic, tamc_portfolio_heuristic.name, 6, 5)
