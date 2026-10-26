# 04 — â-versus-a POD (signal-response POD)

Plain-language summary of the â-vs-a approach described in MIL-HDBK-1823A (the handbook is the
authority). Used for the **amplitude-based rules engine**, whose output is a continuous signal
`â` (peak echo amplitude), not just hit/miss.

## 1. Model
For flaw size `a` and measured response `â`:

    ln â = β₀ + β₁ ln a + ε,     ε ~ Normal(0, σ²)

## 2. Censoring
Responses below the recording/noise floor `â_low` are only known to be `≤ â_low` (left-censored);
responses at saturation `â_high` are only known to be `≥ â_high` (right-censored). Ordinary least
squares would be biased, so we maximise the censored (Tobit) likelihood:

    ℓ = Σ_uncensored ln[ φ(r_i)/σ ] + Σ_left ln Φ(r_i) + Σ_right ln[1 − Φ(r_i)],
    r_i = (ln â_i − β₀ − β₁ ln a_i) / σ

with `φ`, `Φ` the standard normal density and CDF.

## 3. From the signal model to POD
A flaw is detected when `â > â_dec` (the decision threshold). Then

    POD(a) = P(ln â > ln â_dec) = Φ( (β₀ + β₁ ln a − ln â_dec) / σ ).

This is a cumulative normal in `ln a` with mean `μ = (ln â_dec − β₀)/β₁` and spread `σ/β₁`, so

    a50 = exp(μ),   a90 = exp(μ + 1.2816 σ/β₁).

## 4. a90/95 — delta method
With `θ = (β₀, β₁, ln σ)` and covariance `Σ_θ` (inverse observed information),
`g(θ) = ln a90 = (ln â_dec − β₀)/β₁ + 1.2816 σ/β₁`. Then

    a90/95 = exp( g(θ̂) + 1.645 sqrt(∇gᵀ Σ_θ ∇g) ),

with `∇g` computed analytically or by central finite differences (documented in the code).

## Questions for Ruri
1. Why does censoring matter when many small flaws are below the noise floor?
2. When is hit/miss POD the right choice instead of â-vs-a?
3. How does raising `â_dec` trade POD against false calls?
