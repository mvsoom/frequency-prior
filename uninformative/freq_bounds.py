# Formant frequency bounds for adult males and females [a]
# --------------------------------------------------------
#
# [a]: Anatomic maturation is generally assumed by the age of 20
#      [@Vorperian2019]
#
# We select these fairly loosely from 4 sources:
#
# [P]   @Peterson1952           Raw data
# [H]   @Hillenbrand1995        Table V
# [V]   @Vallee1994
# [K]   @Kent2018               Table 1
# [v]   @Vorperian2019          Figures 1, 2
# [k]   @Klatt1908              Table I
#
# The procedure is to take the min and max (averaged) value and round it to the
# nearest hundredth +/- n*100 Hz, where n is the nth formant. The exception is
# for [V], because these are parameters used for a female vocal tract; here
# we just give the number rounded to the nearest hundredth. In the case of
# [P], we just give the exact min and max value from the raw data for adults.
# In the case of [k], the ranges given are "permitted ranges of values [for
# the formant synthesizer]", so need to be taken with a bit of salt.
#
# Fn    min     max     source
# ----- ------- ------- -------
# F1    200     1000    H
#       200     1000    K
#       200     700     V
#       190     1110    P
#       200     1200    v
#       150     900     k
#
# F2    800     3000    H
#       800     2900    K
#       700     2200    V
#       560     3100    P
#       1000    3500    v
#       500     2500    k
#
# F3    1400    3700    H
#       2200    3500    K
#       2000    3200    V
#       1400    3900    P
#       2000    4000    v
#       1300    3500    k
#
# F4    2900    4700    H
#       3900    3400    V
#       3000    5500    v
#       2500    4500    k
#
# F5    4000    4900    V
#       3500    4900    k
#               5500    [b]
#
# F6    4000    5000    k
#               5500    [b]
#
# [b] An average adult female speaker has a vocal tract length that requires an
#     average ceiling of 5500 Hz (which is Praat's standard value), an average
#     adult male speaker has a vocal tract length that requires an average ceiling
#     of 5000 Hz.
#     <https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html>
def bounds():
    return [(200., 600., 1400., 2900, 3500.), (1100., 3500., 4000., 4500., 5500.)]
