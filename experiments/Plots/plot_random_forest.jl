
using MLJ
using DecisionTree
using Plots
using Statistics
using Printf

function plot_random_forest_tree(portfolio_heuristic::QUBOPortfolio.Heuristic, target_heuristic_name::String, tree_index::Int=1, max_depth::Int=4)
    mach = portfolio_heuristic.portfolio.models[target_heuristic_name]
    native_forest = fitted_params(mach).best_fitted_params.forest
    feature_names = report(mach).best_report.features

    @info "\n === Tree Structure ==="
    feature_names_str = String.(feature_names)
    DecisionTree.print_tree(native_forest.trees[tree_index], max_depth, feature_names=feature_names_str)
end
