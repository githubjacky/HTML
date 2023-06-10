import DelimitedFiles: readdlm
import FLoops: @floop
import PyCall: pyimport
import InvertedIndices: Not
import ProgressMeter: Progress, next!
using Distributions



function read_data(path)
    data = readdlm(path, ' ')[:, begin:end-1]
    for j = axes(data, 2)
        for i in axes(data, 1)
            elem = data[i, j]
            try
                start = collect(findfirst(":", elem))[1]+1
                data[i, j] = parse(Float64, elem[start:end])
            catch
            end
            
        end
    end

    features = Matrix{Float64}(hcat(ones(size(data, 1)), data[:, begin+1:end]))
    label = Vector{Float64}(data[:, begin])
    
    return features, label
end


global libsvm = pyimport("libsvm.svmutil")


function train(y, x; param)
    param = libsvm.svm_parameter(param)
    prob = libsvm.svm_problem(y, x)
    model = libsvm.svm_train(prob, param)

    return model
end


function get_coefficient(model)
    sv_dict = model.get_SV()
    sv = Matrix{Float64}(undef, length(sv_dict), length(sv_dict[1]))
    for i = axes(sv, 1), j = axes(sv, 2)
        sv[i, j] = sv_dict[i][j]
    end
    sv_coef = [i[1] for i in model.get_sv_coef()]

    coef = sv' * sv_coef

    return coef
end


function binary_error(model; y, x)
    _, p_acc, _ = libsvm.svm_predict(y, x, model, "-q")
    err = 1 - (p_acc[1]/100)

    return err
end


struct ValResult{T}
    idx::Int64
    errs::Vector{Float64}
    models::T
end

function pick_param(param, train_y, train_x, eval_y, eval_x)
    models = [
        train(train_y, train_x; param=i)
        for i in param
    ]
    err  = binary_error.(models; y=eval_y, x=eval_x)
    idx = argmin(err)
    @inbounds model = models[idx]

    return ValResult(idx, err, models)
end


function pick_param_val(y, x; param)
    N = size(x, 1)
    pos = sample(1:N, 200, replace=false)
    @inbounds train_x, eval_x = x[Not(pos), :], x[pos, :]
    @inbounds train_y, eval_y = y[Not(pos)], y[pos]
    res = pick_param(param, train_y, train_x, eval_y, eval_x)

    return res
end


"""
features: 
782×17 Matrix{Float64}:
 1.0  -0.333333   -0.333333   -0.333333    0.0666667  -0.733333  -0.466667   -0.2         0.2        -0.733333   -0.0666667  -0.0666667   0.466667   -0.466667   -0.0666667  -0.6         0.466667
 1.0   0.2         0.333333   -0.2         0.866667   -0.2       -0.2         0.2        -0.333333   -0.733333    0.6         0.0666667  -0.0666667  -0.6         0.0666667   0.6        -0.333333
 1.0  -0.466667   -0.0666667  -0.2         0.333333   -0.2        0.333333   -0.333333   -0.466667   -0.466667    0.0666667  -0.6        -0.0666667  -0.733333   -0.2        -0.0666667   0.0666667
 1.0  -0.466667    0.0666667  -0.466667   -0.2        -0.6       -0.6        -0.2        -0.2        -0.6        -0.0666667  -0.0666667   0.466667   -0.6         0.0666667  -0.6         0.466667
 1.0  -0.2         0.2         0.0666667  -0.0666667  -0.2       -0.2        -0.2        -0.866667   -0.0666667   0.333333   -0.0666667   0.333333   -0.6         0.0666667  -0.466667    0.0666667
 1.0  -0.733333   -0.6        -0.6        -0.866667   -0.866667  -0.2        -0.0666667  -0.866667   -0.2         0.333333   -0.0666667   0.333333   -0.6         0.0666667  -0.733333    0.0666667
 1.0  -0.333333   -0.2        -0.0666667  -0.466667   -0.333333  -0.2        -0.0666667  -0.866667   -0.2         0.333333   -0.0666667   0.333333   -0.6         0.0666667  -0.6         0.0666667
 1.0  -0.733333   -0.0666667  -0.6        -0.333333   -0.733333  -0.0666667  -0.0666667  -0.6         0.466667    0.0666667  -0.2         0.0666667  -1.0         0.0666667  -0.0666667   0.0666667
 1.0  -0.466667    0.333333   -0.333333    0.0666667  -0.466667  -0.6        -0.2        -0.0666667  -0.466667   -0.0666667  -0.0666667   0.6        -0.6         0.0666667  -0.6         0.466667
 1.0  -0.466667    0.333333   -0.333333   -0.0666667  -0.733333  -0.6         0.0666667   0.0666667  -0.733333   -0.0666667  -0.333333    0.466667   -0.6         0.0666667  -0.6         0.333333
 1.0  -0.333333   -0.0666667  -0.0666667  -0.333333   -0.333333  -0.333333   -0.0666667  -0.733333   -0.2         0.333333    0.0666667   0.466667   -0.6         0.0666667  -0.6         0.0666667


 features(after sorting):
-Inf   -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf         -Inf        -Inf        -Inf
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.733333   -0.6        -0.866667   -1.0        -0.333333   -0.866667   -0.333333   -1.0         -0.466667   -0.866667   -0.733333
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.6        -0.866667   -1.0        -0.333333   -0.733333   -0.333333   -1.0         -0.466667   -0.866667   -0.6
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.6        -0.866667   -0.866667   -0.333333   -0.733333   -0.333333   -1.0         -0.466667   -0.866667   -0.6
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.333333   -1.0         -0.333333   -0.733333   -0.466667
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.333333
⋮                                                      ⋮                                                           ⋮                                                            ⋮                                                            ⋮                                                          ⋮                                                           ⋮          
 

θ:
782×17 Matrix{Float64}:
-Inf   -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf        -Inf         -Inf        -Inf        -Inf
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.666667   -0.6        -0.866667   -1.0        -0.333333   -0.8        -0.333333   -1.0         -0.466667   -0.866667   -0.666667
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.6        -0.866667   -0.933334   -0.333333   -0.733333   -0.333333   -1.0         -0.466667   -0.866667   -0.6
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.533334   -0.866667   -0.866667   -0.266667   -0.733333   -0.333333   -1.0         -0.4        -0.8        -0.533334
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.266667   -1.0         -0.333333   -0.733333   -0.466667
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.466667
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.4
  1.0   -0.866667   -1.0        -0.866667   -1.0        -1.0        -0.6        -0.466667   -0.866667   -0.866667   -0.2        -0.733333   -0.2        -1.0         -0.333333   -0.733333   -0.333333
  ⋮                                                      ⋮                                                           ⋮                                                            ⋮         
 """

check_sign(a) = a == 0. ? -1. : sign(a)
predict(s::Real, x::Real, θ::Real) = s * check_sign(x - θ)
predict(s::Real, x::AbstractVector, θ::Real) = predict.(Ref(s), x, Ref(θ))
function decision_stump(features, labels)
    n, k = size(features)
    temp = vcat(  # (n+1) * k
        reshape(fill(-Inf, k), 1, k),
        reduce(hcat, sort.(eachcol(features)))
    )
    @inbounds θ = (temp[begin+1:end, :] + temp[begin:end-1, :]) / 2

    pred = Array{Float64}(undef, n, n*k, 2)
    for (κ, s) = enumerate([-1, 1])
        @floop for (j, θⱼ) = enumerate(θ)
            d = ceil(Int64, j/n)
            pred[:, j, κ] = predict(s, features[:, d], θⱼ)
        end 
    end
    incorrect_ptr = pred .!= labels

    return incorrect_ptr, pred, θ
end


function ada_boosting(features, labels; T=1000)
    incorrect_ptr, pred, θ = decision_stump(features, labels)
    
    # model(decision stump) parameters
    d = Vector{Int64}(undef, T)
    theta = Vector{Float64}(undef, T)
    s = Vector{Float64}(undef, T)
 

    # aggegation weights for each sub-model(decision stump)
    alpha = Vector{Float64}(undef, T)

    n = size(pred, 1)
    loss = Vector{Float64}(undef, T)
    best_pred = Matrix{Float64}(undef, n, T)  # best prediction for each iterations
    uₜ = fill(1/n, n)  # scaling factor for each iterations

    p = Progress(T, desc="Boosting: ", color=:white, barlen=30)
    @inbounds for t = eachindex(s, d, theta, alpha)
        lossₜ = dropdims(mean(incorrect_ptr, dims=1); dims=1)
        weighted_lossₜ = dropdims(mean(uₜ  .* incorrect_ptr, dims=1); dims=1)
        
        idx = argmin(weighted_lossₜ)
        loss[t] = lossₜ[idx]
        best_pred[:, t] = pred[:, idx]
        
        d[t] = ceil(Int64, idx[1]/n)
        theta[t] = θ[idx[1]]
        s[t] = [-1, 1][idx[2]]

        opt_correct_ptr = best_pred[:, t] .== labels
        opt_incorrect_ptr = best_pred[:, t] .!= labels

        ϵₜ = (opt_incorrect_ptr' * uₜ) / sum(uₜ)
        diamondₜ = sqrt((1-ϵₜ)/ϵₜ)
        alpha[t] = log(diamondₜ)
        
        uₜ = (opt_incorrect_ptr .* uₜ) * diamondₜ + 
             (opt_correct_ptr .* uₜ) / diamondₜ
        next!(p)
    end
    param = (s=s, d=d, theta=theta, alpha=alpha)

    return check_sign.(best_pred * alpha), loss, param
end


function predict(param::NamedTuple, features)
    s, d, theta, alpha = values(param)
    weighted_pred = Matrix{Float64}(undef, size(features, 1), length(alpha))

    for j = axes(weighted_pred, 2)
        weighted_pred[:, j] = predict(s[j], features[:, d[j]], theta[j])
    end

    return check_sign.(weighted_pred * alpha)
end