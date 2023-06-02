module DecisionStump
export simulate_xy, insample_error, fit, test, read_data

using Distributions
import DelimitedFiles: readdlm


struct model
    s::Float64
    θ::Float64
    E_in::Float64
end
direction(a::model) = getproperty(a, :s)
stump(a::model) = getproperty(a, :θ)


check_sign(a) = a == 0. ? -1. : sign(a)

# return scalar
predict(s::Real, x::Real, θ::Real) = s * check_sign(x - θ)
# dim: length(x) * 1
predict(s::Real, x::AbstractVector, θ::Real) = predict.(Ref(s), x, Ref(θ))
# dim: length(x) * length(θ)
predict(s::Real, x::AbstractVector, θ::Vector) = reduce(
    hcat, 
    predict.(Ref(s), Ref(x), θ)
)
# dim: length(x) * length(s)
predict(s::Vector, x::AbstractVector, θ::Real) = reduce(
    hcat, 
    predict.(s, Ref(x), Ref(θ))
)


function (a::model)(x::AbstractVector, y::Vector)
    pred = predict(direction(a), x, stump(a))
    error = mean(pred .!= y)
    # print("error: $(error)")
    return error
end
(a::model)(x::AbstractMatrix, y::Vector, i) = a(x[:, i], y)


function simulate_xy(size; θ=0., τ=0., s=1.)
    x = rand(Uniform(-0.5, 0.5), size)
    y = predict(s, x, θ)  # dim: size*1
    
    idx = findall(
        x->x==1, 
        rand(Binomial(1, τ), size)
    )
    y[idx] = -1. * y[idx]  # flip
    
    return x, y
end


function insample_error(x::Vector, y::Vector, θ::Real)
    pred = predict([-1., 1.], x, θ)  # dim: length(x)*2
    error = [mean(i .!= y) for i=eachcol(pred)]

    return error  # dim: 2*1
end
# dim: 2*length(θ)
insample_error(x::Vector, y::Vector, θ::Vector) = reduce(
    hcat, 
    insample_error.(Ref(x), Ref(y), θ)
)
insample_error(a::model) = getproperty(a, :E_in)

outsample_error(s, θ, τ) = 
    s == 1. ? 
    minimum([abs(θ), 0.5])*(1-2τ)+τ : (1-minimum([abs(θ), 0.5]))*(1-2τ)+τ

find_s(idx) = isodd(idx) ? -1. : 1.

function fit(_x::AbstractVector, _y::Vector)
    indices = sortperm(_x)
    x, y, N = _x[indices], _y[indices], length(_x)

    temp1 = zeros(N+1); temp1[2:end] = x
    temp2 = zeros(N+1); temp2[1:end-1] = x
    θ = ((temp1+temp2) ./ 2)[begin+1:end-1]
    push!(θ, -Inf)

    E_in = insample_error(x, y, θ)  # dim: 2*N
    indices = findall(x->x==minimum(E_in), E_in)
    idx = indices[
        argmin([
            find_s(i[1]) * θ[i[2]] 
            for i in indices
        ])
    ]

    res = model(
        find_s(idx[1]),
        θ[idx[2]],
        E_in[idx]
    )

    return res
end
fit(_x::Matrix, _y::Vector) = fit.(eachcol(_x), Ref(_y))


function test(n=10000; k, τ)
    res = zeros(n)
    for i = eachindex(res)
        hypothesis = fit(simulate_xy(k; τ=τ)...)
        E_in = insample_error(hypothesis)
        # x_test, y_test = simulate_xy(100000)
        # E_out = hypothesis(x_test, y_test)
        E_out = outsample_error(
            direction(hypothesis), 
            stump(hypothesis), 
            τ
        )
        res[i] = E_out - E_in
    end
    
    return mean(res)
end


function read_data(path)
    data = readdlm(path, '\t', Float64, '\n')
    features = data[:, begin:end-1]
    label = data[:, end]
    
    return features, label
end


end  # end of module