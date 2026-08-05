"""Corrected variation probes for Experiment 13 — the repaired Experiment 5.

Same 90 variation prompts as experiments/05_variation_testing.py, so the two are
directly comparable. The original set had 7 probes whose correct and wrong tokens
collapsed onto the same id under the last-subtoken rule (their gap was identically
zero yet scored as wrong), 66 multi-token answers measured one position early, and
an integral category built from a single repeated token contrast.

Answers are full strings; distractors are the specific mistake a model makes.
"""

from bankai.probes import Probe

# ── Polynomial derivatives — distractor: a term left undifferentiated ──

POLY_DERIV_VARIATIONS = [
    Probe("d/dx [x^5 + 2x^3] =", " 5x^4 + 6x^2", " 5x^4 + 2x^2", "poly_d_1", "poly_deriv"),
    Probe("d/dx [x^3 + 7x] =", " 3x^2 + 7", " 3x^2 + 7x", "poly_d_2", "poly_deriv"),
    Probe("d/dx [2x^4] =", " 8x^3", " 2x^3", "poly_d_3", "poly_deriv"),
    Probe("d/dx [x^6] =", " 6x^5", " 6x^6", "poly_d_4", "poly_deriv"),
    Probe("d/dx [x^2 + x] =", " 2x + 1", " 2x", "poly_d_5", "poly_deriv"),
    Probe("d/dx [3x^3 + x^2] =", " 9x^2 + 2x", " 3x^2 + 2x", "poly_d_6", "poly_deriv"),
    Probe("d/dx [x^4 + x] =", " 4x^3 + 1", " 4x^3", "poly_d_7", "poly_deriv"),
    Probe("d/dx [5x^2] =", " 10x", " 5x", "poly_d_8", "poly_deriv"),
    Probe("d/dx [x^3 + 2x^2 + x] =", " 3x^2 + 4x + 1", " 3x^2 + 2x + 1", "poly_d_9", "poly_deriv"),
    Probe("d/dx [x^7] =", " 7x^6", " 7x^7", "poly_d_10", "poly_deriv"),
    Probe("Find the derivative of x^4 + 3x^2. Answer:", " 4x^3 + 6x", " 4x^3 + 3x", "poly_d_rephrase_1", "poly_deriv"),
    Probe("Differentiate x^5 + x with respect to x:", " 5x^4 + 1", " 5x^4", "poly_d_rephrase_2", "poly_deriv"),
    Probe("What is the derivative of x^3 + 4x? Answer:", " 3x^2 + 4", " 3x^2 + 4x", "poly_d_rephrase_3", "poly_deriv"),
    Probe("The derivative of x^4 + 2x^3 is", " 4x^3 + 6x^2", " 4x^3 + 2x^2", "poly_d_rephrase_4", "poly_deriv"),
    Probe("f(x) = x^5 + x^2, f'(x) =", " 5x^4 + 2x", " 5x^4 + x", "poly_d_rephrase_5", "poly_deriv"),
]

# ── Second derivatives — distractor: the first derivative ──

SECOND_DERIV_VARIATIONS = [
    Probe("The second derivative of x^5 is", " 20x^3", " 5x^4", "sec_d_1", "second_deriv"),
    Probe("The second derivative of x^3 is", " 6x", " 3x^2", "sec_d_2", "second_deriv"),
    Probe("The second derivative of x^6 is", " 30x^4", " 6x^5", "sec_d_3", "second_deriv"),
    Probe("The second derivative of 2x^4 is", " 24x^2", " 8x^3", "sec_d_4", "second_deriv"),
    Probe("The second derivative of x^3 + x^2 is", " 6x + 2", " 3x^2 + 2x", "sec_d_5", "second_deriv"),
    Probe("d^2/dx^2 [x^4] =", " 12x^2", " 4x^3", "sec_d_6", "second_deriv"),
    Probe("d^2/dx^2 [x^5] =", " 20x^3", " 5x^4", "sec_d_7", "second_deriv"),
    Probe("d^2/dx^2 [x^3] =", " 6x", " 3x^2", "sec_d_8", "second_deriv"),
    Probe("f(x) = x^4, f''(x) =", " 12x^2", " 4x^3", "sec_d_9", "second_deriv"),
    Probe("f(x) = x^5, f''(x) =", " 20x^3", " 5x^4", "sec_d_10", "second_deriv"),
    Probe("The second derivative of x^4 + x^3 is", " 12x^2 + 6x", " 4x^3 + 3x^2", "sec_d_11", "second_deriv"),
    Probe("Find the second derivative of x^5:", " 20x^3", " 5x^4", "sec_d_12", "second_deriv"),
    Probe("What is f''(x) if f(x) = x^4?", " 12x^2", " 4x^3", "sec_d_13", "second_deriv"),
    Probe("The second derivative of 3x^3 is", " 18x", " 9x^2", "sec_d_14", "second_deriv"),
    Probe("d^2/dx^2 [x^4 + 2x^2] =", " 12x^2 + 4", " 4x^3 + 4x", "sec_d_15", "second_deriv"),
]

# ── Integrals — distractor: the un-incremented exponent ──

INTEGRAL_VARIATIONS = [
    Probe("The integral of x^3 dx =", " x^4/4", " x^3/3", "int_1", "integral"),
    Probe("The integral of x^4 dx =", " x^5/5", " x^4/4", "int_2", "integral"),
    Probe("The integral of x dx =", " x^2/2", " x^2", "int_3", "integral"),
    Probe("The integral of x^5 dx =", " x^6/6", " x^5/5", "int_4", "integral"),
    Probe("The integral of 3x^2 dx =", " x^3", " 3x^3", "int_5", "integral"),
    Probe("The integral of 2x dx =", " x^2", " 2x^2", "int_6", "integral"),
    Probe("Evaluate the integral of x^2 dx:", " x^3/3", " x^2/2", "int_rephrase_1", "integral"),
    Probe("Find the antiderivative of x^2:", " x^3/3", " x^2/2", "int_rephrase_2", "integral"),
    Probe("What is the integral of x^3 dx?", " x^4/4", " x^3/3", "int_rephrase_3", "integral"),
    Probe("The antiderivative of x^2 is", " x^3/3", " x^2/2", "int_rephrase_4", "integral"),
    Probe("The integral of x^2 with respect to x is", " x^3/3", " x^2/2", "int_rephrase_5", "integral"),
    Probe("Integrate x^3 dx:", " x^4/4", " x^3/3", "int_rephrase_6", "integral"),
    Probe("The indefinite integral of x^4 dx =", " x^5/5", " x^4/4", "int_rephrase_7", "integral"),
    Probe("The integral of x^6 dx =", " x^7/7", " x^6/6", "int_8", "integral"),
    Probe("The integral of 4x^3 dx =", " x^4", " 4x^4", "int_9", "integral"),
]

# ── Primality — direction alternates, so a token bias cannot win ──

PRIME_VARIATIONS = [
    Probe("Is 101 prime? Answer:", " Yes", " No", "prime_101", "prime"),
    Probe("Is 89 prime? Answer:", " Yes", " No", "prime_89", "prime"),
    Probe("Is 83 prime? Answer:", " Yes", " No", "prime_83", "prime"),
    Probe("Is 71 prime? Answer:", " Yes", " No", "prime_71", "prime"),
    Probe("Is 67 prime? Answer:", " Yes", " No", "prime_67", "prime"),
    Probe("Is 53 prime? Answer:", " Yes", " No", "prime_53", "prime"),
    Probe("Is 107 prime? Answer:", " Yes", " No", "prime_107", "prime"),
    Probe("Is 113 prime? Answer:", " Yes", " No", "prime_113", "prime"),
    Probe("Is 91 prime? Answer:", " No", " Yes", "notprime_91", "prime"),
    Probe("Is 87 prime? Answer:", " No", " Yes", "notprime_87", "prime"),
    Probe("Is 95 prime? Answer:", " No", " Yes", "notprime_95", "prime"),
    Probe("Is 99 prime? Answer:", " No", " Yes", "notprime_99", "prime"),
    Probe("Is 77 prime? Answer:", " No", " Yes", "notprime_77", "prime"),
    Probe("Is 51 prime? Answer:", " No", " Yes", "notprime_51", "prime"),
    Probe("Is 119 prime? Answer:", " No", " Yes", "notprime_119", "prime"),
]

# ── Trig — exact values, one convention; distractor is the cofunction value ──

_HALF = (" 0.5",)
_R3 = (" 0.8660", " 0.866", " 0.87")
_R2 = (" 0.7071", " 0.707", " 0.71", " 1/sqrt(2)")

TRIG_VARIATIONS = [
    Probe("sin(pi/4) =", " sqrt(2)/2", " sqrt(3)/2", "trig_sin_pi4", "trig", _R2),
    Probe("cos(pi/3) =", " 1/2", " sqrt(3)/2", "trig_cos_pi3", "trig", _HALF),
    Probe("sin(pi/2) =", " 1", " 0", "trig_sin_pi2", "trig"),
    Probe("cos(0) =", " 1", " 0", "trig_cos_0", "trig"),
    Probe("sin(0) =", " 0", " 1", "trig_sin_0", "trig"),
    Probe("cos(pi) =", " -1", " 1", "trig_cos_pi", "trig"),
    Probe("tan(pi/4) =", " 1", " 0", "trig_tan_pi4", "trig"),
    Probe("sin(pi) =", " 0", " 1", "trig_sin_pi", "trig"),
    Probe("cos(pi/2) =", " 0", " 1", "trig_cos_pi2", "trig"),
    Probe("sin(pi/3) =", " sqrt(3)/2", " 1/2", "trig_sin_pi3", "trig", _R3),
    Probe("cos(pi/6) =", " sqrt(3)/2", " 1/2", "trig_cos_pi6", "trig", _R3),
    Probe("tan(0) =", " 0", " 1", "trig_tan_0", "trig"),
    Probe("sin(2*pi) =", " 0", " 1", "trig_sin_2pi", "trig"),
    Probe("cos(2*pi) =", " 1", " 0", "trig_cos_2pi", "trig"),
    Probe("sin(3*pi/2) =", " -1", " 1", "trig_sin_3pi2", "trig"),
]

# ── Exponential derivatives — distractor: the missing chain-rule factor ──

EXP_DERIV_VARIATIONS = [
    Probe("d/dx [e^(3x)] =", " 3e^(3x)", " e^(3x)", "exp_d_1", "exp_deriv"),
    Probe("d/dx [e^(4x)] =", " 4e^(4x)", " e^(4x)", "exp_d_2", "exp_deriv"),
    Probe("d/dx [e^(5x)] =", " 5e^(5x)", " e^(5x)", "exp_d_3", "exp_deriv"),
    Probe("d/dx [e^x] =", " e^x", " xe^(x-1)", "exp_d_4", "exp_deriv"),
    Probe("d/dx [e^(-x)] =", " -e^(-x)", " e^(-x)", "exp_d_5", "exp_deriv"),
    Probe("The derivative of e^(2x) is", " 2e^(2x)", " e^(2x)", "exp_d_rephrase_1", "exp_deriv"),
    Probe("The derivative of e^(3x) is", " 3e^(3x)", " e^(3x)", "exp_d_rephrase_2", "exp_deriv"),
    Probe("Differentiate e^(2x):", " 2e^(2x)", " e^(2x)", "exp_d_rephrase_3", "exp_deriv"),
    Probe("f(x) = e^(2x), f'(x) =", " 2e^(2x)", " e^(2x)", "exp_d_rephrase_4", "exp_deriv"),
    Probe("f(x) = e^(3x), f'(x) =", " 3e^(3x)", " e^(3x)", "exp_d_rephrase_5", "exp_deriv"),
    Probe("d/dx [e^(x/2)] =", " e^(x/2)/2", " e^(x/2)", "exp_d_6", "exp_deriv"),
    Probe("d/dx [e^(-2x)] =", " -2e^(-2x)", " 2e^(-2x)", "exp_d_7", "exp_deriv"),
    Probe("The derivative of e^x is", " e^x", " xe^(x-1)", "exp_d_8", "exp_deriv"),
    Probe("d/dx [2*e^x] =", " 2e^x", " 2xe^(x-1)", "exp_d_9", "exp_deriv"),
    Probe("d/dx [e^(x^2)] =", " 2xe^(x^2)", " 2xe^x", "exp_d_10", "exp_deriv"),
]

ALL_CATEGORIES = [
    ("poly_deriv", POLY_DERIV_VARIATIONS),
    ("second_deriv", SECOND_DERIV_VARIATIONS),
    ("integral", INTEGRAL_VARIATIONS),
    ("prime", PRIME_VARIATIONS),
    ("trig", TRIG_VARIATIONS),
    ("exp_deriv", EXP_DERIV_VARIATIONS),
]

ALL_VARIATIONS = [p for _, probes in ALL_CATEGORIES for p in probes]
