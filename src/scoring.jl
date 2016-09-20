### subroutines used to compute scores and score tests
### those marked with a leading underscore '_' are not exported

export glm_score_test!, glm_score_test

"""
Compute the score and observed information matrix for the residual sum of squares loss.
Arguments:

- `s` is a preallocated vector to store the **s**core.
- `F` is a preallocated matrix to store the Fischer in**f**ormation matrix.
- `r` is a preallocated vector to store the **r**esiduals.
- `w` is a preallocated vector to store the Poisson **w**eights.
- `X` is the `n \times p` design matrix.
- `y` is the `n`-dimensional response.
- `β` is the `p`-dimensional parameter vector.

Output:

`s` and `F`, with the calculated score and Fisher information matrix.
"""
function _compute_linear_score!{T <: AbstractFloat}(
    s :: Vector{T}, # s = X' * (y - X*β)
    F :: Matrix{T}, # F = -X' * X
    r :: Vector{T}, # r = y - X*β
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T},
)
    # compute residuals r = y - X*β
    BLAS.gemv!('N', -one(T), X, β, zero(T), r)
    BLAS.axpy!(one(T), y,  r)

    # compute the score s =  X' * r = X' * (y - X*β)
    BLAS.gemv!('T', one(T), X, r, zero(T), s)

    # compute the information matrix W = X' * X
    BLAS.gemm!('T', 'N', -one(T), X, X, zero(T), F)

    return (s, F)
end

"""
Compute the Poisson score as part of a score test.

Arguments:

- `s` is a preallocated vector to store the **s**core.
- `F` is a preallocated matrix to store the Fischer in**f**ormation matrix.
- `r` is a preallocated vector to store the **r**esiduals.
- `w` is a preallocated vector to store the Poisson **w**eights.
- `X` is the `n \times p` design matrix.
- `y` is the `n`-dimensional response.
- `β` is the `p`-dimensional parameter vector.

Output:

`s` and `F`, with the calculated score and Fisher information matrix.
"""
function _compute_poisson_score!{T <: AbstractFloat}(
    s :: Vector{T}, # *s*core
    F :: Matrix{T}, # in*F*ormation matrix
    r :: Vector{T}, # r = y - exp(X*β)
    w :: Vector{T}, # *w*eight vector
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T}
)
    # compute residuals r = y - exp(X*β) piecemeal
    # first store r = exp(X*β), save those weights to w, then add y to r
    BLAS.gemv!('N', one(T), X, β, zero(T), r)
    exp!(r)
    copy!(w, r)
    BLAS.axpy!(-one(T), y, r) # r = - y + r

    # compute the score s = - X' * r = X' * (y - X*β)
    BLAS.gemv!('T', -one(T), X, r, zero(T), s)

    # apply weights to X
    sqrt!(w)
    scale!(w, X) # X = diagm(w) * X

    # compute information matrix F = X' * diag(w) * X
    BLAS.gemm!('T', 'N', one(T), X, X, zero(T), F)

    # unscale X
    reciprocal!(w)
    scale!(w, X) # now X = 1 ./ diagm(w) * X, back to original copy

    return (s, F)
end

"""
Compute the logistic score as part of a score test.

Arguments:

- `s` is a preallocated vector to store the **s**core.
- `F` is a preallocated matrix to store the Fischer in**f**ormation matrix.
- `r` is a preallocated vector to store the **r**esiduals.
- `w` is a preallocated vector to store the Poisson **w**eights.
- `X` is the `n \times p` design matrix.
- `y` is the `n`-dimensional response.
- `β` is the `p`-dimensional parameter vector.

Output:

`s` and `F`, with the calculated score and Fisher information matrix.
"""
function _compute_logistic_score!{T <: AbstractFloat}(
    s :: Vector{T}, # *s*core
    F :: Matrix{T}, # in*F*ormation matrix
    r :: Vector{T}, # r = y - exp(X*β)
    w :: Vector{T}, # *w*eight vector
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T}
)
    # first store estimated response r = X*β
    BLAS.gemv!('N', one(T), X, β, zero(T), r)
   
    # must transform the response with the logistic link
    # function gives r = 1 ./ (1 .+ exp(-r))
    logistic!(r)

    # save proper logistic weights w = r .* (1 .- r)
    logistic_weights!(w, r)

    # finish computing residuals r = y - logistic(X*β)
    BLAS.axpy!(-one(T), y, r) # r = r - y = logistic(Xβ) - y 

    # compute the score s = - X' * r = X' * (y - logistic(X*β))
    BLAS.gemv!('T', -one(T), X, r, zero(T), s)

    # transform weights for scaling of X
    # here w = sqrt(w)
    sqrt!(w)

    # X = diagm(w) * X scales data matrix with weights; will undo this later
    scale!(w, X)

    # compute the observed information matrix F = X' * W * X
    BLAS.gemm!('T', 'N', one(T), X, X, zero(T), F)

    # need to undo weight scaling
    # entails computing w = 1 ./ w and rescaling X back to original copy
    reciprocal!(w)
    scale!(w, X) # now X = 1 ./ diagm(w) * X, back to original copy

    return (s, F)
end

function glm_score_test!{T <: AbstractFloat}(
    s :: Vector{T},
    F :: Matrix{T},
    r :: Vector{T},
    w :: Vector{T},
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T},
    model::ASCIIString
)
    # error handling
    (n, p) = size(X)
    @assert p == length(s)
    @assert (p,p) == size(F)

    # handle each model separately
    model == "linear"   && _compute_linear_score!(s, F, r, X, y, β)
    model == "logistic" && begin
        ##clamp!(r, -20*one(T), 20*one(T))
        _compute_logistic_score!(s, F, r, w, X, y, β)
    end
    model == "Poisson" && begin
        ##clamp!(r, -20*one(T), 20*one(T))
        _compute_poisson_score!(s, F, r, w, X, y, β)
    end

    # compute the score statistic from the score and information.
    z = F \ s
    score_test = dot(s, z)

    return score_test
end # function glm_score_test!

"""
    glm_score_test(X, y, β, model) -> Float 

Performs a score test for either linear, logistic, or Poisson regression
with a canonical link function.
   
Arguments:

- `X` is the design matrix
- `y` is the response vector
- `β` is the MLE under the null hypothesis

Output:

The floating point value of the score.
"""
function glm_score_test{T <: AbstractFloat}(
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T},
    model::ASCIIString
)
    # error handling
    if model ∉ ("linear", "logistic", "Poisson")
        throw(ArgumentError(
          "The only model choices are linear, logistic, and Poisson.\n \n"))
    end
    (n, p) = size(X)
    @assert n == length(y)
    @assert p == length(β)

    # allocate temporary arrays
    score       = zeros(T, p)
    information = zeros(T, p, p)
    residuals   = zeros(T, n)
    weights     = zeros(T, n)

    # compute score test
    score_test = glm_score_test!(score, information, residuals, weights, X, y, β, model)

    return score_test
end # function glm_score_test
