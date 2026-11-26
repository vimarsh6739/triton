; p = probability at a valid column, v = its seed, c = a row constant.
; Softmax's symmetric Jacobian maps a seed to p*(v - dot(probabilities,seed)).
; Let s be that dot product and mass be the sum of valid probabilities.
; Linearity makes the centered seed's dot product s - c*mass.
; Both the JVP and VJP use this identity. Evaluation in floating point can differ.
(set-logic QF_NRA)
(declare-const p Real)
(declare-const v Real)
(declare-const s Real)
(declare-const c Real)
(declare-const mass Real)
(assert (= mass 1))
(define-fun original () Real (* p (- v s)))
(define-fun centered () Real (* p (- (- v c) (- s (* c mass)))))
(assert (distinct original centered))
(check-sat)
