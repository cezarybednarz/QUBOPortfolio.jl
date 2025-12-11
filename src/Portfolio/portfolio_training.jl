

RandomForestClassifier = MLJ.@load RandomForestClassifier pkg=DecisionTree verbosity=0
XGBoostClassifier = MLJ.@load XGBoostClassifier pkg=XGBoost verbosity=0

function train_test_split(data, target_col::Symbol, train_ratio=0.7)
    # Ensure target is categorical with consistent levels (false/true) so predicted probability vectors match
    raw_y = data[:, target_col]
    y = categorical(raw_y, ordered=false, levels=[false, true])

    X = data[:, Not(target_col)]

    n = size(data, 1)
    indices = shuffle(1:n)
    train_size = Int(floor(train_ratio * n))

    train_indices = indices[1:train_size]
    test_indices = indices[train_size+1:end]

    X_train = X[train_indices, :]
    y_train = y[train_indices]

    X_test = X[test_indices, :]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test
end


function _create_tuned_random_forest_model(X_train::DataFrame)
    # "setting ntree, the number of trees to build, to 500 (..)
    # nodesize the minimum number of observations in leaves, to 1"
    model = RandomForestClassifier(n_trees=500, min_samples_leaf=1)

    # "(the mtry parameter) using five-fold cross-validation on the training set"
    # `mtry` corresponds to `n_subfeatures` in the model
    tuning_range = MLJ.range(model, :n_subfeatures, lower=1, upper=size(X_train, 2))
    tuning_param = :n_subfeatures
    resampling_strategy = MLJBase.StratifiedCV(nfolds=5, shuffle=true)

    tuned_model = MLJ.TunedModel(
        model=model,
        resampling=resampling_strategy,
        range=tuning_range,
        tuning=MLJ.Grid(resolution=5),
        measures=[MLJ.accuracy],
        train_best=true,
    )

    return tuned_model
end

function _create_tuned_xgboost_model(X_train::DataFrame)
    model = XGBoostClassifier(
        num_round=200, # Increased number of rounds
        objective="binary:logistic" # More appropriate for binary classification
    )

    # Define a more extensive hyperparameter tuning space
    tuning_space = [
        MLJ.range(model, :max_depth, lower=3, upper=10),
        MLJ.range(model, :eta, lower=1e-2, upper=0.3, scale=:log), # Learning rate
        MLJ.range(model, :subsample, lower=0.6, upper=1.0), # Row sampling
        MLJ.range(model, :colsample_bytree, lower=0.6, upper=1.0), # Column sampling
        MLJ.range(model, :gamma, lower=0, upper=5), # Regularization
        MLJ.range(model, :min_child_weight, lower=1, upper=10) # Regularization
    ]

    resampling_strategy = MLJBase.StratifiedCV(nfolds=5, shuffle=true)

    # Use RandomSearch for more efficient exploration of the larger parameter space
    tuned_model = MLJ.TunedModel(
        model=model,
        resampling=resampling_strategy,
        range=tuning_space,
        tuning=MLJ.Grid(resolution=2), # 5-fold cross-validation
        measures=[MLJ.log_loss], # Log loss is often better for probabilities
        train_best=true,
    )

    return tuned_model
end

function train!(
    algorithm_portfolio::AlgorithmPortfolio,
    execution_data::Dict{String,DataFrame},
    target_function::QUBOPortfolio.TargetFunction=QUBOPortfolio.HIGHEST_MEAN,
    classifier::Symbol=:random_forest,
)
    for (heuristic, data) in execution_data
        # train all (100%) data
        X_train, y_train, X_test, y_test = train_test_split(data, target_function.name, 1.0)

        # Create and tune the model
        if classifier == :random_forest
            tuned_model = _create_tuned_random_forest_model(X_train)
        elseif classifier == :xgboost
            tuned_model = _create_tuned_xgboost_model(X_train)
        else
            error("Unknown classifier type: $classifier. Use :random_forest or :xgboost")
        end

        # Create a machine with the tuned model and fit it
        machine = MLJ.machine(tuned_model, X_train, y_train)
        @info " Fitting machine for $heuristic. Train size: $(size(X_train, 1)), Test size: $(size(X_test, 1))."
        MLJ.fit!(machine)

        # Save the trained machine to the portfolio
        algorithm_portfolio.models[heuristic] = machine
    end
end
