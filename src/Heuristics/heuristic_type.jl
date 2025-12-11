@enum HeuristicType begin
    # check all solutions O(2^N)
    BRUTEFORCE
    # https://github.com/USCqserver/tamc
    TAMC
    # https://github.com/JuliaQUBO/MQLib.jl
    MQLIB
    # https://github.com/jmejia8/Metaheuristics.jl
    METAHEURISTICSJL_PSO
    METAHEURISTICSJL_BRKGA
    # https://github.com/bqth29/simulated-bifurcation-algorithm
    SIMULATED_BIFURCATION
    # This is a wrapper for the QUBOPortfolio module (this module)
    QUBOPORTFOLIO
end

# Abstract type for AlgorithmPortfolio for Heuristic type to break circular dependency
abstract type AbstractAlgorithmPortfolio end

Base.@kwdef struct Heuristic
    # Heuristic type
    type::HeuristicType
    # Heuristic name (for MQLib this is the name we use to load the model, e.g. "BURER2002")
    name::String=""
    hyperparameters::Dict{String, Any} = Dict()
    # If this Heuristic is an algorithm portfolio
    portfolio::Union{QUBOPortfolio.AbstractAlgorithmPortfolio, Nothing}=nothing
end
