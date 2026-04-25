# Transcript: coefficient-interpretation WITH skill

## Task
The logistic regression coefficient for 'Disease Duration' is -1.58. What does that mean?

## Steps

1. Read SKILL.md — Step 6 explains: coefficients are log-odds on the standardised scale; positive = increases probability of positive class; negative = decreases it. Use for ranking feature importance, not absolute clinical magnitude.
2. Read references/metrics-guide.md — Odds ratio section: `OR = exp(coef)`. OR < 1 → negative/protective association. One-unit increase in standardised feature multiplies odds by exp(coef).
3. Applied: exp(-1.58) ≈ 0.206. Direction: negative (Disease Duration decreases probability of positive class). Magnitude: large (high absolute value = high feature importance rank).
4. Noted the standardisation caveat: "one unit" = one standard deviation on the original scale.
5. Wrote structured response with odds ratio table and caveats.

## Conclusions
Disease Duration = -1.58 (log-odds, standardised). Odds ratio ≈ 0.206, meaning each SD increase in Disease Duration reduces odds by ~79%. Strong negative predictor.
