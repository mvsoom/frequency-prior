% Discovered errors in original `ana` code

`$ cd $WRK/proj/ana/notebook/nested-sampling`

1. `model.py:186`: The scaling factors are not necessary, since $Z$
   is the expected value of the likelihood over the prior. We sample
   from the unit hypercube anyway, and no corrections are being made
   for that either.

2. `model.py:78`: A log() needs to be taken of `χ2_total`.

These errors did not affect the estimates of $B$ and $F$.