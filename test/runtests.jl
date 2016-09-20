using MendelBase
using Base.Test

# write your own tests here

# linear regression
X = [68., 49, 60, 68, 97, 82, 59, 50, 73, 39, 71, 95, 61, 72, 87, 
40, 66, 58, 58, 77]
y = [75., 63, 57, 88, 88, 79, 82, 73, 90, 62, 70, 96, 76, 75, 85,
40, 74, 70, 75, 72]
X = [ones(size(X,1)) X]
(estimate, loglikelihood) = Regress.regress(X, y, "linear") # Jennrich section 1.4
@test_approx_eq estimate [-34.56756756756755,-0.600487705750864]
@test_approx_eq loglikelihood -51.87044390678082 

# logistic regression
X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75,
3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
y = [0., 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
X = [ones(size(X,1)) X]
(estimate, loglikelihood) = Regress.regress(X, y, "logistic") # Wikipedia problem
@test_approx_eq estimate [-4.077713366075042,1.504645401369091] 
@test_approx_eq loglikelihood -8.029878464344675 


# poisson regression
X = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
X = reshape(X, 14, 1)
y = [0., 1 ,2 ,3, 1, 4, 9, 18, 23, 31, 20, 25, 37, 45]
println("poisson")
(estimate, loglikelihood) = Regress.regress(X, y, "Poisson") # Aids problem
estimate = [0.0; estimate]
X = [ones(size(X,1)) X]
test = Regress.glm_score_test(X, y, estimate, "Poisson")
(estimate, loglikelihood) = Regress.regress(X, y, "Poisson") # Aids problem
@test_approx_eq estimate [0.3396339220518644,0.25652359361235305] 
@test_approx_eq loglikelihood 472.0625479329488 
