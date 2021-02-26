####################################################
# Occam factors (see Jaynes 2003 for a discussion) #
####################################################
source("run_stats.R")

# Conclusion:
# Not so thrustworthy because (log L)_max (i.e. logl) values are not
# close enough to each other for a faithful decomposition of log Z into
# (log L)_max and log W (i.e. Occam factor). Better to just compare
# the log Z values as their errors should be smaller due to the
# averaging.
occ = full_cmp[, .(logw = logz - logl, logl, logz), key=.(vowel, P, Q, new)]

docc = occ[, .(
  delta_logw = .SD[new==T, logw] - .SD[new==F, logw],
  delta_logl = .SD[new==T, logl] - .SD[new==F, logl],
  delta_logz = .SD[new==T, logz] - .SD[new==F, logz]
), key = .(vowel, P, Q)]

ggplot(occ[Q==5], aes(y=logw, color=new, shape=new)) +
  geom_point(aes(P)) +
  facet_wrap(~vowel)

ggplot(docc[Q==5], aes(y=delta_logw)) +
  geom_point(aes(P)) +
  facet_wrap(~vowel)

ggplot(docc[Q==5], aes(y=delta_logl)) +
  geom_point(aes(P)) +
  facet_wrap(~vowel)

ggplot(docc[Q==5], aes(y=delta_logz)) +
  geom_point(aes(P)) +
  facet_wrap(~vowel)