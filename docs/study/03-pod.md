# 03 — Probability of detection (POD): hit/miss and a90/95

Approach follows the hit/miss method described in MIL-HDBK-1823A. The handbook itself is the
authority; this note is a plain-language summary for study, not a copy of it.

## 1. Data
Each flawed specimen `i` has a size `a_i` and an outcome `y_i ∈ {0, 1}` (missed / found).

## 2. Model
A logistic GLM on log size:

    POD(a) = 1 / (1 + exp(−(β₀ + β₁ ln a)))

Fit `β = (β₀, β₁)` by maximum likelihood:

    ℓ(β) = Σ_i [ y_i ln p_i + (1 − y_i) ln(1 − p_i) ],   p_i = POD(a_i).

## 3. a50 and a90
Solve `POD(a_p) = p`:

    ln a_p = (logit(p) − β₀) / β₁,   logit(p) = ln(p / (1 − p)).

So `a50 = exp(−β₀/β₁)` and `a90 = exp((ln 9 − β₀)/β₁)`.

## 4. a90/95 (the 95 % upper confidence bound on a90) — delta method
Let `Σ` be the covariance of `β̂` (the inverse of the observed Fisher information).
For `g(β) = ln a90 = (L − β₀)/β₁` with `L = logit(0.9)`:

    ∇g = ( −1/β₁ ,  −(L − β₀)/β₁² )
    Var[ln a90] ≈ ∇gᵀ Σ ∇g
    a90/95 = exp( ln â90 + 1.645 · sqrt(Var[ln a90]) )      (one-sided 95 %)

The handbook also describes likelihood-ratio bounds, which behave better with few data.
We implement the delta method and say so in every report.

## 5. False-call rate
On unflawed specimens: `FCR = (number flagged) / (number unflawed)`. With zero false calls in
`n` images, the 95 % upper bound on the rate is about `3/n` (rule of three) — why "0 false calls
on 100 images" is weak evidence (bound ≈ 3 %).

## 6. Why ~30+ flaws
With few independent flaws the covariance `Σ` is large, so `a90/95` is far above `a90`.
Augmented copies of the **same** physical crack are not independent flaws.

## Questions for Ruri
1. Why log size and not size?
2. Derive the gradient `∇g` above.
3. Why can a model have a90 = 1 mm but a90/95 = 3 mm?
4. What does the rule of three say about 0 false calls in 200 images?
