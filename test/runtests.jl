using Test

include("Metric/metric.jl")
include("Utils/convert.jl")
include("Heuristics/heuristic.jl")
include("Portfolio/test_portfolio_evaluation.jl")

function test_main()
    @testset "QUBOPortfolio test suite" verbose = true begin
        test_heuristic()
        test_metric()
        test_convert()
        test_portfolio_evaluation()
    end
end

test_main()
