###################################
# Comparing the old vs. new prior #
###################################
source("run_stats.R")

# p(P,Q | vowel)
PQ_posterior = full_posterior[, .(prob = sum(p)), key = .(vowel, P, Q)]

ggplot(
  PQ_posterior
) +
  geom_text(aes(P, Q, label=pct(prob)), size=3, color="black") +
  coord_equal() +
  scale_fill_gradient2() +
  facet_wrap(~vowel, nrow=2)

######################################################
# Comparing models at same (P,Q) for new=F and new=T #
######################################################

# Differences are (property value of new model) - (property value of old model)
setkey(full, "vowel", "P", "Q", "new")
diff = full[, .(
  d_niter = diff(niter),
  d_logl = diff(logl),
  d_logz = diff(logz),
  d_information = diff(information),
  d_walltime = diff(walltime)
), key = .(vowel, P, Q)]

diff = diff[PQ_posterior]

# The new prior emphasizes the regions of higher likelihood better
# (hence higher log Z) and enables the sampler to find higher log
# likelihood points; this is true for virtually all cases.
#
# So for a given order (P, Q), the new prior almost always achieves
# better performance.
ggplot(diff, aes(d_logz, d_logl)) +
  geom_point(aes(alpha=prob, size=prob, fill=vowel, shape=vowel)) +
  scale_color_brewer(type="qual") +
  scale_alpha(range=c(.1,1)) +
  scale_size_continuous(range=c(2,10)) +
  ggtitle("The new prior gives rise to higher evidence and SNR at the ML estimate in almost all runs")

# For a given order (P,Q), while the new prior has very often
# better log L and log Z, the information is typically comparable,
# also for the best models.
summary(diff[, .(d_niter, d_logl, d_logz, d_information, d_walltime)])

ggplot(diff, aes(d_logz, d_information)) +
  geom_point(aes(alpha=prob, size=prob, fill=vowel, shape=vowel)) +
  scale_color_brewer(type="qual") +
  scale_alpha(range=c(.1,1)) +
  scale_size_continuous(range=c(2,10)) +
  ggtitle("The information between old and new priors for given (P,Q) is typically comparable")

############################################
# Comparing MAP models for new=F and new=T #
############################################

# Get the MAP models given new to compare their performance
MAP_new = full[, .(P, Q,
                   best = logz == max(logz),
                   logl,
                   logz,
                   information,
                   joint_prob = normalize(exp(-(max(logz) - logz)))), # p(new,P,Q|vowel)
               by=.(vowel, new)]

MAP_best = MAP_new[best==T]
setkey(MAP_best, "vowel", "new")

MAP_best[, log_occam := logz - logl]

MAP_diff = MAP_best[, .(
  d_logz = diff(logz),
  d_logl = diff(logl),
  d_log_occam = diff(log_occam),
  d_information = diff(information)
), key = .(vowel)]

# Note that the likelihood, Z, information and Occam factors
# can be compared across models with different model orders
# because their units stay consistent. The units of the likelihood
# function are (amplitude unit)^(-N), where N is the number of data points.
MAP_best
MAP_diff

############################################
# Conclusions
############################################
#
# - The best model is always one with the new prior.
#
# - The best model given the old prior has substantially
#   different model order than the best model given the new prior.
#   This explains some of the big differences in MAP_diff.
#
# - For a given model order (P, Q), the new prior almost
#   always results in higher Z and likelihood at ML (see code above),
#   and in a comparable information gain. The new prior is thus
#   about as informative as the Jeffreys prior, but with some
#   extra desirable properties added:
#
#   - Less and easier prior parameters to tune for the user
#   - Consistency in ordering and no overlap
#   - Able of guiding the sampler to higher log L and thus
#     higher log Z values
#
# When comparing the best models given new=F and new=T we find:
#
# - The new prior allows much better model performance, yielding
#   higher MAP values of Q and higher SNR, both for given (P,Q)
#   and for MAP models.
#
# - Thus, log Z is much greater for the new prior, indicating that the
#   new prior places its mass in better regions of high likelihood;
#   or, equivalently, that the data was much more probable under
#   the new prior;
#
# - log L is much greater, indicating that the best models according
#   to log Z also achieve much higher SNR at their ML estimates;
#   
# - The Occam factor "is the amount by which the model is penalized
#   by our nonoptimal prior information" (Jaynes 2003 p. 604). This
#   is seen to be smaller for the new MAP models. (Note that it is
#   dimensionless.) Thus the new MAP models have "smaller amount of
#   prior probability in the high-likelihood region picked out
#   by the data", indicating their larger spread.
#
# - The amount of information gained by going from the prior to the
#   posterior using the MAP models is also higher with the new prior
#   (but note both posterior and prior change depending on the model).
#   Thus the new prior favours models with posteriors for which it
#   is more uninformative.
#
# Current hypothesis:
# The new prior allowed the model to place VTRs essentially where-ever
# it wanted, leading to higher SNRs as an rather unexpected outcome
# of a more uninformative prior. The value of the new prior is that
# it elegantly solves the problem of allowing VTRs to place themselves
# naturally along the frequency axis which is much harder with
# bounded priors such as the old prior without wasting much
# computing power (e.g. specifying U(200, 5500) priors for each VTR,
# which leads to strong multimodality).