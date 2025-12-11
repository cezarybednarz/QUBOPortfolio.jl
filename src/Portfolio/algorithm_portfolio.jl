
Base.@kwdef struct AlgorithmPortfolio <: AbstractAlgorithmPortfolio
    heuristics::Vector{QUBOPortfolio.Heuristic} = Vector{QUBOPortfolio.Heuristic}()
    models::Dict{String, MLJ.Machine} = Dict{String, MLJ.Machine}()
    cached_metrics::Union{Dict{String, Vector{Metric}}, Nothing} = nothing
end

function create_portfolio(heuristics::Vector{QUBOPortfolio.Heuristic})
    cached_metrics = QUBOPortfolio.load_metrics_from_cache("cache/metrics.jld2")
    return AlgorithmPortfolio(heuristics=heuristics, cached_metrics=cached_metrics)
end
