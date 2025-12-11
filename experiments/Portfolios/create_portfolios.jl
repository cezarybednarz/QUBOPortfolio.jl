if !isdefined(Main, :QUBOPortfolio)
    include("../../src/QUBOPortfolio.jl")
end
using JLD2

function create_and_train_portfolio_heuristic(
                                    heuristics::Vector{QUBOPortfolio.Heuristic},
                                    training_instances::Vector{<:QUBOPortfolio.AbstractInstance},
                                    name::String;
                                    cache::QUBOPortfolio.ResultCache=nothing,
                                    top_k::Int=1,
                                    type::Symbol=:random_forest,
                                    use_portfolio_cache::Bool=true,
    )::QUBOPortfolio.Heuristic

    if use_portfolio_cache
        filename = "cache/portfolios/$(name)_heuristic.jld2"
        if isfile(filename)
            @load filename portfolio_heuristic
            @info "Loaded portfolio heuristic '$(name)' from cache."
            return portfolio_heuristic
        end
    end
    repeat_rate = 3
    target_function = QUBOPortfolio.HIGHEST_MEAN
    training_results = QUBOPortfolio.run_heuristics_on_dataset(
        heuristics,
        training_instances;
        repeat=repeat_rate,
        verbosity=true,
        cache=cache,
        readonly_cache=true,
        max_concurrent_tasks=8 # TODO adjust based on system
    )
    cached_metrics = QUBOPortfolio.load_metrics_from_cache("cache/metrics.jld2")
    classified_instances = QUBOPortfolio.classify_dataset_with_cache(training_instances, cached_metrics)
    selector_training_data = QUBOPortfolio.create_selector_training_data(
        heuristics,
        classified_instances,
        training_results;
        target_function=target_function,
        repeat=repeat_rate
    )
    portfolio = QUBOPortfolio.create_portfolio(heuristics)
    QUBOPortfolio.train!(portfolio, selector_training_data, target_function, type)
    portfolio_heuristic = QUBOPortfolio.Heuristic(
        type=QUBOPortfolio.QUBOPORTFOLIO,
        name=name,
        portfolio=portfolio,
        hyperparameters=Dict("top_k" => top_k)
    )
    if use_portfolio_cache
        JLD2.@save "cache/portfolios/$(name)_heuristic.jld2" portfolio_heuristic
    end
    return portfolio_heuristic
end
