# Transcript: coefficient-interpretation WITHOUT skill

## Task
The logistic regression coefficient for 'Disease Duration' is -1.58. What does that mean?

## Steps

1. Recalled logistic regression theory: coefficients are on the log-odds scale. β means for a one-unit increase in the predictor, the log-odds of the outcome change by β.
2. Interpreted sign: negative (-1.58) → Disease Duration is inversely associated with the outcome.
3. Converted to odds ratio: exp(-1.58) ≈ 0.206 → each unit increase reduces odds to ~20.6%.
4. Assessed magnitude: -1.58 is a large coefficient, indicating substantial effect.
5. Noted caveats: unit of measurement matters; statistical significance must be checked separately.

## Conclusions
Disease Duration is a strong negative predictor. Odds ratio ≈ 0.206 (~79% decrease in odds per unit increase).
