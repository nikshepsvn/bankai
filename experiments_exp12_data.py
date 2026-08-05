"""Corrected probe data for Experiment 12 — the repaired version of Experiment 6.

Same 90 prompts as experiments_exp6_data.py, so results are directly comparable.
Three defects in the original set are fixed:

  1. Answers are now full strings, not single token ids. The original relied on
     encode_token()'s last-subtoken rule, under which " 20" and " 0" both reduce
     to "0" — 8 of the 90 probes had correct_id == wrong_id and a logit gap that
     was identically zero regardless of any weight flip.

  2. Distractors are per-probe and plausible, not a constant " 0". In the original
     set the hardcoded wrong token was not the model's top competitor on 76 of 90
     probes (median rank 23), so most gaps measured distance to a token the model
     was not considering. Here the distractor is the specific mistake a model
     actually makes: the first derivative where the second is asked for, the
     un-incremented exponent on an integral, the cofunction value in trig.

  3. The integral category had one contrast (" 1" vs " 2") repeated 15 times, so
     it contributed a single measurement while being counted as a category. Each
     integral probe now carries its own answer and its own distractor.

Trig answers use exact values in consistent ASCII form throughout. The original
mixed conventions — sin(pi/6) was scored against the fraction form "1/2" while
sin(pi/4) was scored against the decimal form "0.707" — so a flip that helped
one necessarily hurt the other.
"""

from bankai.probes import Probe


# ── Polynomial derivatives — distractor: a dropped or un-differentiated term ──

POLY_DERIV_TRAIN = [
    Probe("d/dx [x^4 + 3x^2] =", " 4x^3 + 6x", " 4x^3 + 3x", "pd_train_0", "poly_deriv"),
    Probe("d/dx [x^5 + 2x^3] =", " 5x^4 + 6x^2", " 5x^4 + 2x^2", "pd_train_1", "poly_deriv"),
    Probe("d/dx [x^3 + 7x] =", " 3x^2 + 7", " 3x^2 + 7x", "pd_train_2", "poly_deriv"),
    Probe("d/dx [2x^4] =", " 8x^3", " 2x^3", "pd_train_3", "poly_deriv"),
    Probe("d/dx [x^6] =", " 6x^5", " 6x^6", "pd_train_4", "poly_deriv"),
    Probe("d/dx [x^2 + x] =", " 2x + 1", " 2x", "pd_train_5", "poly_deriv"),
    Probe("d/dx [3x^3 + x^2] =", " 9x^2 + 2x", " 3x^2 + 2x", "pd_train_6", "poly_deriv"),
    Probe("d/dx [x^3 + 2x^2 + x] =", " 3x^2 + 4x + 1", " 3x^2 + 2x + 1", "pd_train_7", "poly_deriv"),
    Probe("The derivative of x^4 + 2x^3 is", " 4x^3 + 6x^2", " 4x^3 + 2x^2", "pd_train_8", "poly_deriv"),
    Probe("Differentiate x^5 + x with respect to x:", " 5x^4 + 1", " 5x^4", "pd_train_9", "poly_deriv"),
]
POLY_DERIV_VAL = [
    Probe("d/dx [x^7 + x] =", " 7x^6 + 1", " 7x^6", "pd_val_0", "poly_deriv"),
    Probe("d/dx [4x^3] =", " 12x^2", " 4x^2", "pd_val_1", "poly_deriv"),
    Probe("d/dx [x^4 + x^3 + x^2] =", " 4x^3 + 3x^2 + 2x", " 4x^3 + x^2 + x", "pd_val_2", "poly_deriv"),
    Probe("Find the derivative of 2x^5 + x:", " 10x^4 + 1", " 10x^4", "pd_val_3", "poly_deriv"),
    Probe("f(x) = x^6 + 3x, f'(x) =", " 6x^5 + 3", " 6x^5 + 3x", "pd_val_4", "poly_deriv"),
]

# ── Second derivatives — distractor: the first derivative ──

SECOND_DERIV_TRAIN = [
    Probe("The second derivative of x^4 is", " 12x^2", " 4x^3", "sd_train_0", "second_deriv"),
    Probe("The second derivative of x^5 is", " 20x^3", " 5x^4", "sd_train_1", "second_deriv"),
    Probe("The second derivative of x^3 is", " 6x", " 3x^2", "sd_train_2", "second_deriv"),
    Probe("d^2/dx^2 [x^4] =", " 12x^2", " 4x^3", "sd_train_3", "second_deriv"),
    Probe("d^2/dx^2 [x^5] =", " 20x^3", " 5x^4", "sd_train_4", "second_deriv"),
    Probe("d^2/dx^2 [x^3] =", " 6x", " 3x^2", "sd_train_5", "second_deriv"),
    Probe("f(x) = x^4, f''(x) =", " 12x^2", " 4x^3", "sd_train_6", "second_deriv"),
    Probe("The second derivative of x^6 is", " 30x^4", " 6x^5", "sd_train_7", "second_deriv"),
    Probe("The second derivative of 2x^4 is", " 24x^2", " 8x^3", "sd_train_8", "second_deriv"),
    Probe("Find the second derivative of x^5:", " 20x^3", " 5x^4", "sd_train_9", "second_deriv"),
]
SECOND_DERIV_VAL = [
    Probe("d^2/dx^2 [x^7] =", " 42x^5", " 7x^6", "sd_val_0", "second_deriv"),
    Probe("The second derivative of x^3 + x^2 is", " 6x + 2", " 3x^2 + 2x", "sd_val_1", "second_deriv"),
    Probe("f(x) = x^5, f''(x) =", " 20x^3", " 5x^4", "sd_val_2", "second_deriv"),
    Probe("The second derivative of 3x^3 is", " 18x", " 9x^2", "sd_val_3", "second_deriv"),
    Probe("What is f''(x) if f(x) = x^6?", " 30x^4", " 6x^5", "sd_val_4", "second_deriv"),
]

# ── Integrals — distractor: the un-incremented exponent ──

INTEGRAL_TRAIN = [
    Probe("The integral of x^2 dx =", " x^3/3", " x^2/2", "int_train_0", "integral"),
    Probe("The integral of x^3 dx =", " x^4/4", " x^3/3", "int_train_1", "integral"),
    Probe("The integral of x^4 dx =", " x^5/5", " x^4/4", "int_train_2", "integral"),
    Probe("The integral of x dx =", " x^2/2", " x^2", "int_train_3", "integral"),
    Probe("The integral of x^5 dx =", " x^6/6", " x^5/5", "int_train_4", "integral"),
    Probe("Evaluate the integral of x^2 dx:", " x^3/3", " x^2/2", "int_train_5", "integral"),
    Probe("Find the antiderivative of x^2:", " x^3/3", " x^2/2", "int_train_6", "integral"),
    Probe("The antiderivative of x^3 is", " x^4/4", " x^3/3", "int_train_7", "integral"),
    Probe("Integrate x^4 dx:", " x^5/5", " x^4/4", "int_train_8", "integral"),
    Probe("The indefinite integral of x^3 dx =", " x^4/4", " x^3/3", "int_train_9", "integral"),
]
INTEGRAL_VAL = [
    Probe("The integral of x^6 dx =", " x^7/7", " x^6/6", "int_val_0", "integral"),
    Probe("The integral of x^7 dx =", " x^8/8", " x^7/7", "int_val_1", "integral"),
    Probe("What is the integral of x^2 dx?", " x^3/3", " x^2/2", "int_val_2", "integral"),
    Probe("The antiderivative of x^4 is", " x^5/5", " x^4/4", "int_val_3", "integral"),
    Probe("The integral of x^2 with respect to x is", " x^3/3", " x^2/2", "int_val_4", "integral"),
]

# ── Primality — direction alternates, so a token-bias shortcut cannot win ──

PRIME_TRAIN = [
    Probe("Is 97 prime? Answer:", " Yes", " No", "pr_train_0", "prime"),
    Probe("Is 101 prime? Answer:", " Yes", " No", "pr_train_1", "prime"),
    Probe("Is 89 prime? Answer:", " Yes", " No", "pr_train_2", "prime"),
    Probe("Is 83 prime? Answer:", " Yes", " No", "pr_train_3", "prime"),
    Probe("Is 71 prime? Answer:", " Yes", " No", "pr_train_4", "prime"),
    Probe("Is 91 prime? Answer:", " No", " Yes", "pr_train_5", "prime"),
    Probe("Is 87 prime? Answer:", " No", " Yes", "pr_train_6", "prime"),
    Probe("Is 95 prime? Answer:", " No", " Yes", "pr_train_7", "prime"),
    Probe("Is 67 prime? Answer:", " Yes", " No", "pr_train_8", "prime"),
    Probe("Is 99 prime? Answer:", " No", " Yes", "pr_train_9", "prime"),
]
PRIME_VAL = [
    Probe("Is 107 prime? Answer:", " Yes", " No", "pr_val_0", "prime"),
    Probe("Is 113 prime? Answer:", " Yes", " No", "pr_val_1", "prime"),
    Probe("Is 53 prime? Answer:", " Yes", " No", "pr_val_2", "prime"),
    Probe("Is 77 prime? Answer:", " No", " Yes", "pr_val_3", "prime"),
    Probe("Is 119 prime? Answer:", " No", " Yes", "pr_val_4", "prime"),
]

# ── Trig — exact values, one convention; distractor is the cofunction value ──

TRIG_TRAIN = [
    Probe("sin(pi/6) =", " 1/2", " sqrt(3)/2", "trig_train_0", "trig", (" 0.5",)),
    Probe("sin(pi/2) =", " 1", " 0", "trig_train_1", "trig"),
    Probe("cos(0) =", " 1", " 0", "trig_train_2", "trig"),
    Probe("sin(0) =", " 0", " 1", "trig_train_3", "trig"),
    Probe("cos(pi) =", " -1", " 1", "trig_train_4", "trig"),
    Probe("tan(pi/4) =", " 1", " 0", "trig_train_5", "trig"),
    Probe("sin(pi) =", " 0", " 1", "trig_train_6", "trig"),
    Probe("cos(pi/2) =", " 0", " 1", "trig_train_7", "trig"),
    Probe("tan(0) =", " 0", " 1", "trig_train_8", "trig"),
    Probe("cos(2*pi) =", " 1", " 0", "trig_train_9", "trig"),
]
TRIG_VAL = [
    Probe("sin(pi/4) =", " sqrt(2)/2", " sqrt(3)/2", "trig_val_0", "trig",
          (" 0.7071", " 0.707", " 0.71", " 1/sqrt(2)", " 1/2 sqrt(2)")),
    Probe("cos(pi/3) =", " 1/2", " sqrt(3)/2", "trig_val_1", "trig", (" 0.5",)),
    Probe("sin(pi/3) =", " sqrt(3)/2", " 1/2", "trig_val_2", "trig", (" 0.8660", " 0.866", " 0.87")),
    Probe("cos(pi/6) =", " sqrt(3)/2", " 1/2", "trig_val_3", "trig", (" 0.8660", " 0.866", " 0.87")),
    Probe("sin(3*pi/2) =", " -1", " 1", "trig_val_4", "trig"),
]

# ── Exponential derivatives — distractor: the missing chain-rule factor ──

EXP_DERIV_TRAIN = [
    Probe("d/dx [e^(2x)] =", " 2e^(2x)", " e^(2x)", "ed_train_0", "exp_deriv"),
    Probe("d/dx [e^(3x)] =", " 3e^(3x)", " e^(3x)", "ed_train_1", "exp_deriv"),
    Probe("d/dx [e^(4x)] =", " 4e^(4x)", " e^(4x)", "ed_train_2", "exp_deriv"),
    Probe("d/dx [e^x] =", " e^x", " xe^(x-1)", "ed_train_3", "exp_deriv"),
    Probe("d/dx [e^(-x)] =", " -e^(-x)", " e^(-x)", "ed_train_4", "exp_deriv"),
    Probe("The derivative of e^(2x) is", " 2e^(2x)", " e^(2x)", "ed_train_5", "exp_deriv"),
    Probe("The derivative of e^(3x) is", " 3e^(3x)", " e^(3x)", "ed_train_6", "exp_deriv"),
    Probe("f(x) = e^(2x), f'(x) =", " 2e^(2x)", " e^(2x)", "ed_train_7", "exp_deriv"),
    Probe("Differentiate e^(2x):", " 2e^(2x)", " e^(2x)", "ed_train_8", "exp_deriv"),
    Probe("The derivative of e^x is", " e^x", " xe^(x-1)", "ed_train_9", "exp_deriv"),
]
EXP_DERIV_VAL = [
    Probe("d/dx [e^(5x)] =", " 5e^(5x)", " e^(5x)", "ed_val_0", "exp_deriv"),
    Probe("f(x) = e^(3x), f'(x) =", " 3e^(3x)", " e^(3x)", "ed_val_1", "exp_deriv"),
    Probe("d/dx [e^(-2x)] =", " -2e^(-2x)", " 2e^(-2x)", "ed_val_2", "exp_deriv"),
    Probe("d/dx [2*e^x] =", " 2e^x", " 2xe^(x-1)", "ed_val_3", "exp_deriv", (" 2 e^x",)),
    Probe("d/dx [e^(x^2)] =", " 2xe^(x^2)", " 2xe^x", "ed_val_4", "exp_deriv"),
]

ALL_TRAIN = (
    POLY_DERIV_TRAIN + SECOND_DERIV_TRAIN + INTEGRAL_TRAIN +
    PRIME_TRAIN + TRIG_TRAIN + EXP_DERIV_TRAIN
)

ALL_VAL = (
    POLY_DERIV_VAL + SECOND_DERIV_VAL + INTEGRAL_VAL +
    PRIME_VAL + TRIG_VAL + EXP_DERIV_VAL
)

VAL_BY_CATEGORY = [
    ("poly_deriv", POLY_DERIV_VAL),
    ("second_deriv", SECOND_DERIV_VAL),
    ("integral", INTEGRAL_VAL),
    ("prime", PRIME_VAL),
    ("trig", TRIG_VAL),
    ("exp_deriv", EXP_DERIV_VAL),
]
