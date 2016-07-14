################################################################################
# This set of functions provides various genetic utilities.
################################################################################
#
# External modules.
#
using Distributions # From package Distributions.

export map_function, inverse_map_function
export hardy_weinberg_test, xlinked_hardy_weinberg_test, simes_fdr

"""
  This function calculates recombination fractions based on
  Haldane's or Kosambi's formula.
"""
function map_function(d::Float64, choice::ASCIIString)

  if choice == "Haldane"
    theta = 0.5 * (1.0 - exp(-2d))
  elseif choice == "Kosambi"
    theta = 0.5 * tanh(2d)
  else
  throw(ArgumentError(
    "Only Haldane's or Kosambi's map function is allowed.\n \n"))
  end
  return theta
end # function map_function

"""
This function calculates genetic map distances based on Haldane's or
Kosambi's formula.
"""
function inverse_map_function(theta::Float64, choice::ASCIIString)

  if choice == "Haldane"
    if theta >= 0.5
      d = Inf
    else
      d = -0.5 * log(1.0 - 2.0 * theta)
    end
  elseif choice == "Kosambi"
    if theta >= 0.5
      d = Inf
    else
      d = 0.25 * log((1.0 + 2.0 * theta) / (1.0 - 2.0 * theta))
    end
  else
    throw(ArgumentError(
      "Only Haldane's or Kosambi's map function is allowed.\n \n"))
  end
  return d
end # function inverse_map_function

"""
This function tests for Hardy-Weinberg equilibrium at a SNP.
The genotype vector conveys for each person his/her number of
reference alleles. Thus, all genotypes belong to {0, 1, 2, NaN}.
The pvalue of the chi-square statistic is returned.
"""
function hardy_weinberg_test(genotype::Vector{Float64})

  (expected, observed) = (zeros(3), zeros(3))
  #
  # Tally observations per cell.
  #
  n = 0
  for i = 1:length(genotype)
    if !isnan(genotype[i])
      n = n + 1
      g = round(Int, genotype[i])
      j = g + 1
      observed[j] = observed[j] + 1.0
    end
  end
  if n == 0
    return 1.0
  end
  #
  # Compute allele frequencies.
  #
  p = (2.0 * observed[1] + observed[2]) / (2 * n)
  if p <= 0.0 || p >= 1.0
    return 1.0
  else
    q = 1.0 - p
    #
    # Sum over cells of the contingency table.
    #
    expected[1] = n * p^2
    expected[2] = n * 2.0 * p * q
    expected[3] = n * q^2
    test = 0.0
    for i = 1:3
    test = test + (observed[i] - expected[i])^2 / expected[i]
  end
  return ccdf(Chisq(1), test)
  end
end # function hardy_weinberg_test

"""
This function tests for Hardy-Weinberg equilibrium at an
xlinked SNP. The genotype vector conveys for each person his/her
number of reference alleles. Thus, all genotypes belong to {0, 1, 2, NaN}.
The pvalue of the chisquare statistic is returned.
"""
function xlinked_hardy_weinberg_test(genotype::Vector{Float64}, 
  male::BitArray{1})

  (expected_female, observed_female) = (zeros(3), zeros(3))
  (expected_male, observed_male) = (zeros(2), zeros(2))
  #
  # Tally observations per cell and count members of each sex.
  #
  (females, males) = (0, 0)
  for i = 1:length(genotype)
    if !isnan(genotype[i])
      g = round(Int, genotype[i])
      if male[i]
        males = males + 1
        j = min(g + 1, 2)
        observed_male[j] = observed_male[j] + 1.0
      else
        females = females + 1
        j = g + 1
        observed_female[j] = observed_female[j] + 1.0
      end
    end
  end
  if females + males == 0
    return 1.0
  end
  #
  # Compute allele frequencies.
  #
  p = 2.0 * observed_female[1] + observed_female[2] + observed_male[1]
  p = p / (2 * females + males)
  if p <= 0.0 || p >= 1.0
    return 1.0
  end
  q = 1.0 - p
  #
  # Sum over female cells of the contingency table.
  #
  expected_female[1] = females * p^2
  expected_female[2] = females * 2.0 * p * q
  expected_female[3] = females * q^2
  test = 0.0
  for i = 1:3
    term = (observed_female[i] - expected_female[i])^2 / expected_female[i]
    test = test + term
  end
  #
  # Sum over male cells of the contingency table.
  #
  expected_male[1] = males * p
  expected_male[2] = males * q
  for i = 1:2
    term = (observed_male[i] - expected_male[i])^2 / expected_male[i]
    test = test + term
  end
  return ccdf(Chisq(2), test)
end # function xlinked_hardy_weinberg_test

"""
This function implements the Simes false discovery rate (FDR) procedure
discussed by Benjamini and Hochberg. All p-values at or below the threshold
are declared significant for the given FDR rate and number of tests.
"""
function simes_fdr(pvalue::Vector{Float64}, fdr::Vector{Float64}, tests::Int)

  n = length(fdr)
  threshold = zeros(n)
  number_passing = zeros(Int, n)
  perm = sortperm(pvalue)
  k = 1 # previous value of i
  for j = 1:n
    i = 0
    for i = k:length(pvalue)
      if tests * pvalue[perm[i]] > i * fdr[j] break end
    end
    if i == 1 continue end
      k = i - 1
      threshold[j] = pvalue[perm[i - 1]]
      number_passing[j] = i - 1
  end
  return (number_passing, threshold)
end # function simes_fdr

