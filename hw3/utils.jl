import DelimitedFiles: readdlm
using Distributions, Random

function read_data(path)
    data = readdlm(path, '\t', Float64, '\n')
    features = hcat(ones(size(data, 1)), data[:, begin:end-1]) 
    label = data[:, end]
    
    return features, label
end


mean_squared_error(X, y, w) = mean((X*w-y).^2)

logistic(x) = 1 / (1+exp(-x))
cross_entropy_error(X, y, w) = -mean(log.(logistic.(y .* X*w)))

check_sign(a) = a == 0. ? 1. : sign(a)
binary_error(X, y, w) = mean(check_sign.(X*w) .!= y)


function LS_estimator(X, y)
    # faster: w = X \ y
    w = inv(transpose(X)*X) * (transpose(X)*y)

    return w
end

function SGD_estimator(X, y, sg; it=800, η=0.001, seed=20230420, w₀=nothing)
    Random.seed!(seed)
    w = isa(w₀, Nothing) ? zeros(size(X, 2)) : w₀
    idx = sample(1:size(X, 1), it, replace=true)
    for i = 1:it
        Xᵢ, yᵢ = X[idx[i], :], y[idx[i]]
        w -= η * sg(Xᵢ, yᵢ, w)
    end
    
    return w
end

reg_stochastic_gradient(Xᵢ, yᵢ, w) = 2 * (Xᵢ*transpose(Xᵢ)*w - Xᵢ*yᵢ)
regSGD_estimator(X, y; kwargs...) = SGD_estimator(X, y, reg_stochastic_gradient; kwargs...)

logit_stochastic_gradient(Xᵢ, yᵢ, w) = logistic(-yᵢ*transpose(w)*Xᵢ) * (-yᵢ*Xᵢ)
logitSGD_estimator(X, y; kwargs...) = SGD_estimator(X, y, logit_stochastic_gradient; kwargs...)



function polynomial_transform(X; Q)
    d = size(X, 2)-1
    X_ = Matrix{Float64}(undef, size(X, 1), Q*d+1)
    X_[:, begin:d+1] = X; X = X[:, begin+1:end]

    beg = d + 2
    for q = 2:Q
        en = beg + d - 1
        X_[:, beg:en] = X .^ q
        beg = en + 1
    end

    return X_
end