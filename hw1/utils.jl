import DelimitedFiles: readdlm
import Random: seed!
import Distributions: sample
import Statistics: median, mean
import LinearAlgebra: normalize

normalize_row(a) = vcat([transpose(normalize(i, 2)) for i=eachrow(a)]...)

function read_data(path="hw1_train.txt"; x₀=1, scale=1., normalize=false)
    data = readdlm(path, '\t', Float64, '\n')
    features = hcat(x₀*ones(size(data, 1)), data[:, begin:end-1]) ./ scale
    normalize && (features = normalize_row(features))
    label = data[:, end]
    
    return features, label, size(features, 1)
end

check_sign(a) = a == 0. ? (return 1.) : (return sign(a))

finderror(s::Symbol, args...) = finderror(Val{s}(), args...)
finderror(::Val{:single}, features, w, label) = findfirst(
    x->x!=0., 
    check_sign.(features*w) .- label
)
finderror(::Val{:multiple}, features, w, label) = findall(
    x->x!=0., 
    check_sign.(features*w) .- label
)

function PLA(features, label; M, seed=20230525)
    seed!(seed)
    obs, w, iter, m, M = 1:size(features, 1), 
                         zeros(size(features, 2)), 
                         0, 
                         0,
                         ceil(Int, M)
    while !isnothing(m)
        idx = rand(finderror(:multiple, features, w, label))
        w += label[idx]*features[idx, :]
        iter += 1

        idxes = sample(obs, M; replace=true)
        m = finderror(:single, features[idxes, :], w, label[idxes])
    end
    error = length(finderror(:multiple, features, w, label)) / length(obs)

    return w, iter, error
end
