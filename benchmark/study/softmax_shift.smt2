; At a given logit i let w = exp(x_i - m), S = sum_j w_j > 0 and
; T = sum_j w_j v_j. In direction v, letting dm = c adds -c*w to
; the numerator derivative and -c*S to the denominator derivative.
; These are the two quotient-rule numerators; both divide by S^2.
; This proof uses the exact-real exp derivative and linearity of sums.
(set-logic QF_NRA)
(declare-const w Real)
(declare-const S Real)
(declare-const T Real)
(declare-const v Real)
(declare-const c Real)
(assert (> S 0))
(define-fun differentiated () Real (- (* w (- v c) S) (* w (- T (* c S)))))
(define-fun detached () Real (- (* w v S) (* w T)))
(assert (distinct differentiated detached))
(check-sat)
