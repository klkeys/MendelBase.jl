### common mathematical subroutines used in MendelBase

export exp!
export sqrt!
export reciprocal!
export logistic!
export logistic_weights!
export logistic_loglik
export poisson_loglik
export clamp!

"""
    exp!(x)

Compute `x = exp(x)` in-place, overwriting `x`. 
"""
function exp!{T <: AbstractFloat}(
    x :: Vector{T}
)
    @inbounds for i in eachindex(x)
        x[i] = exp(x[i])
    end
    return x
end

"""
    exp!(y, x)

Compute `y = exp(x)`, overwriting `y`.
"""
function exp!{T <: AbstractFloat}(
    y :: Vector{T},
    x :: Vector{T}
)
    @assert length(y) == length(x)
    @inbounds for i in eachindex(x)
        y[i] = exp(x[i])
    end
    return y
end

"""
    sqrt!(x)

Compute `x = sqrt(x)` in-place, overwriting `x`. 
"""
function sqrt!{T <: AbstractFloat}(
    x :: Vector{T}
)
    @inbounds for i in eachindex(x)
        x[i] = sqrt(x[i])
    end
    return x
end

"""
    reciprocal!(x)

Compute `x = 1 ./ x` in-place.
"""
function reciprocal!{T <: AbstractFloat}(
    x :: Vector{T}
)
    @inbounds for i in eachindex(x)
        x[i] = one(T) ./ x[i]
    end
    return x
end

"""
    logistic_weights!(y, x)

Compute `y = x .* (1 .- x)`, overwriting `y`.
"""
function logistic_weights!{T <: AbstractFloat}(
    y :: Vector{T},
    x :: Vector{T}
)
    @assert length(y) == length(x)
    @inbounds for i in eachindex(x)   
        y[i] = x[i] * (one(T) - x[i])
    end
    return y
end

"""
    logistic!(x)

Compute `x = 1 ./ (1 + exp(-x))` in-place.
For regression frameworks with data `X` and parameters `β`, the assumption is that `x = X*β`.
"""
function logistic!{T <: AbstractFloat}(
    x :: Vector{T}
)
    @inbounds for i in eachindex(x)   
        x[i] = one(T) / (one(T) + exp(-x[i]))
    end
    return x
end



"""
    logistic_loglik(z, y) -> Float

Compute the logistic loglikelihood based on arguments `z = X*β` and the response vector `y`.
For regression frameworks with data `X` and parameters `β`, the assumption is that `z = X*β`.
"""
function logistic_loglik{T <: AbstractFloat}(
    z :: Vector{T}, # use to house X*β and logistic(X*β)
    y :: Vector{T}
)
    logistic!(z)
    obj = zero(T)
    @inbounds for i in eachindex(z)
        obj_i = y[i] > 0 ? log(z[i]) : log(one(T) - z[i])
        obj += obj_i
    end
    return obj
end

"""
    poisson_loglik(z, y) -> Float

Compute the Poisson loglikelihood based on arguments `z = X*β` and the response vector `y`.
"""
function poisson_loglik{T <: AbstractFloat}(
    z :: Vector{T},
    y :: Vector{T}
)
    obj = zero(T)
    @inbounds for i in eachindex(z)
        obj += y[i] * z[i] - exp(z[i])
    end 
    return obj
end

"""
    clamp!(x, a, b)

This subroutine thresholds elements of `x` to lie between scalars `a` and `b`.
"""
function clamp!{T <: AbstractFloat}(
    x :: Vector{T},
    a :: T,
    b :: T
)
    @assert a < b
    @inbounds for i in eachindex(x)
        x[i] = min(max(x[i], a), b)
    end
    return x
end
