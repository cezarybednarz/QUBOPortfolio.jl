include("../../src/QUBOPortfolio.jl")

using Test
using DataFrames
using MLJ
using SimpleWeightedGraphs
using MLJModels

function test_portfolio_evaluation()
    # Mock Heuristics
    h1 = QUBOPortfolio.Heuristic(type=QUBOPortfolio.BRUTEFORCE, name="h1")
    h2 = QUBOPortfolio.Heuristic(type=QUBOPortfolio.BRUTEFORCE, name="h2")
    heuristics = [h1, h2]

    # Mock Instances
    instance1 = QUBOPortfolio.get_instance_from_path("test_example1.qh")
    instance2 = QUBOPortfolio.get_instance_from_path("test_example2.qh")
    instances = [instance1, instance2]

    # Mock Model
    DummyModel = @load ConstantClassifier pkg=MLJModels
    model = machine(DummyModel(), DataFrame(x1=[1]), categorical([true])) # Use DummyModel() to instantiate
    fit!(model, verbosity=0)

    # Mock AlgorithmPortfolio
    models = Dict("h1" => model, "h2" => model)
    portfolio = QUBOPortfolio.AlgorithmPortfolio(heuristics, models)

    @testset "create_selector_training_data" begin
        selector_training_data = QUBOPortfolio.create_selector_training_data(portfolio, instances; target_name=:is_best)
        @test length(selector_training_data) == 2
        @test haskey(selector_training_data, "h1")
        @test haskey(selector_training_data, "h2")

        df_h1 = selector_training_data["h1"]
        df_h2 = selector_training_data["h2"]

        @test names(df_h1)[1] == "is_best"
        @test size(df_h1, 2) > 1
        @test size(df_h2, 2) > 1
    end

    @testset "run_portfolio" begin
        results, chosen = QUBOPortfolio.run_portfolio(portfolio, instances, 1)

        @test length(results) == 2
        @test haskey(results, instance1.name)
        @test haskey(results, instance2.name)

        @test length(chosen) == 2
        @test haskey(chosen, instance1.name)
        @test haskey(chosen, instance2.name)

        @test results[instance1.name] == -29.0
        @test results[instance2.name] == -1.0
        @test chosen[instance1.name][1].name == "h1"
        @test chosen[instance2.name][1].name == "h1"

        # Test with top_k=2
        results_k2, chosen_k2 = QUBOPortfolio.run_portfolio(portfolio, instances, 2)
        @test length(chosen_k2[instance1.name]) == 2
    end
end

# test_portfolio_evaluation()
