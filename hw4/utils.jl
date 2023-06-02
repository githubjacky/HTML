import DelimitedFiles: readdlm
import PyCall: pyimport
import InvertedIndices: Not
import Combinatorics: with_replacement_combinations
using Distributions, Random


function read_data(path)
    data = readdlm(path, '\t', Float64, '\n')
    features = hcat(ones(size(data, 1)), data[:, begin:end-1]) 
    label = data[:, end]
    
    return features, label
end


l2_param_transform(x) = 1 / (2x)
l1_param_transform(x) = 1 / (x)


transform(idx; mat) = broadcast(*, eachcol(mat[:, idx])...)
function polynomial_transform(X; Q)
    d = size(X, 2) - 1
    X_ = Matrix{Float64}(undef, size(X, 1), binomial(Q+d, d))
    X_[:, 1:d+1] = X; X = X[:, begin+1:end]

    start = d + 2
    for q = 2:Q
        idx = collect(with_replacement_combinations(1:d, q))
        terminate = start + length(idx) - 1
        X_[:, start:terminate] = reduce(hcat, transform.(idx; mat=X))
        start = terminate + 1
    end

    return X_
end


global liblinear = pyimport("liblinear.liblinearutil")


function train(c; y, x, s=0)
    param = liblinear.parameter("-s $s -c $c -e 0.000001  -q")
    prob = liblinear.problem(y, x)
    model = liblinear.train(prob, param)

    return model
end


function binary_error(model; y, x)
    _, p_acc, _ = liblinear.predict(y, x, model, "-q")
    err = 1 - (p_acc[1]/100)

    return err
end


struct ValResult{T}
    idx::Int64
    errs::Vector{Float64}
    models::T
end

mini_error(a::ValResult) = a.errs[a.idx]
best_model(a::ValResult) = a.models[a.idx]


function pick_param(param, train_y, train_x, eval_y, eval_x)
    models = train.(param; y=train_y, x=train_x)
    err  = binary_error.(models; y=eval_y, x=eval_x)
    idx = argmin(err)
    @inbounds model = models[idx]

    return ValResult(idx, err, models)
end


function pick_param_val(y, x; param)
    N = size(x, 1)
    pos = sample(1:N, 80, replace=false)
    @inbounds train_x, eval_x = x[Not(pos), :], x[pos, :]
    @inbounds train_y, eval_y = y[Not(pos)], y[pos]
    res = pick_param(param, train_y, train_x, eval_y, eval_x)

    return res
end


function pick_param_crossval(y, x; param, K)
    N = size(x, 1); r = Int64(N/K)
    idx, start = repeat(1:N, outer=2), rand(1:N)
    err = Matrix{Float64}(undef, length(param), K)

    @inbounds for j = axes(err, 2)
        terminate = start + r -1
        pos = idx[start:terminate]
        train_x, eval_x = x[Not(pos), :], x[pos, :]
        train_y, eval_y = y[Not(pos)], y[pos]

        err[:, j] = pick_param(param, train_y, train_x, eval_y, eval_x).errs
        start = terminate + 1
    end
    @inbounds param_ecv = mean(err, dims=2)[:, 1]
    idx = argmin(param_ecv)

    return idx, param_ecv[idx]
end


function get_coefficient(model; Q=4, d=10)
    k = binomial(Q+d, d)
    coef = model.get_decfun_coef.(1:k)

    return coef
end