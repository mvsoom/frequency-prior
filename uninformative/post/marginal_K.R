######################################################
# Calculating log Z and H for Q -- integrating out P #
######################################################
source("run_stats.R")

pi_P = 1/11

d = full[experiment %in% c("compare","free"), .(vowel, new, P, Q, logz, information, experiment)]

# Formula for the evidence where L (trend order) is integrated out is
#
# Z(K) = sum_{L=0}^10 Z(K,L).
#
# Z(K) is called logz_Q in the code.
#
# Here Z(L,K) is the evidence calculated from a nested sample run given
# L, K and data. The posterior probability of the trend order, needed
# for the information (see next comment block) is then
#
# P(L|K) = Z(K,L) / Z(K).
d[, log_prob_P := logz - logsum(logz), key = .(experiment,vowel,new,Q)]
d[, prob_P := exp(log_prob_P), key = .(experiment,vowel,new,Q)]

r = d[, .(logz_Q = logsum(logz)), key=.(experiment,vowel,new,Q)]

# Formula for H(P:pi) (where P and pi are the posterior and prior) that
# integrates out L (trend order) and is conditional on K (num frequencies):
#
# H(P:pi) = sum_{L=0}^10 P(L|K) * H[P:pi|K,L]
#         + sum_{L=0}^10 P(L|K) * log[P(L|K)/pi(L|K)].
#
# Here P(L|K) is the normalized posterior probability of trend order L given K
# and H(P:pi|K,L) is the information given K,L. Both are obtained from a
# single nested run for given K, L, and data (vowel). pi(L|K) = 1/11.
#
# H(P:pi) is called information_Q in the code.
dq = d[, .(
  H_param = sum(prob_P*information), # >= 0
  H_P = sum(prob_P*(log_prob_P- log(pi_P))) # >= 0
      ), key=.(experiment,vowel,new,Q)]

dq[, information_Q := H_param + H_P]

# Very well approximated by just taking log Z and H for the MAP (P,Q)
total = dq[r]

d[, MAP := logz == max(logz), key = .(experiment,vowel,new,Q)]
d[MAP==T]

# Sleek
final = total[, .(experiment,vowel, new, K=Q, H = information_Q, logz = logz_Q)]
final[, p := exp(logz - logsum(logz)), key = .(experiment,new,vowel)]

write.csv(final, 'marginal_K.csv', quote=F, row.names=F)

###################
# Model selection #
###################
ggplot(final, aes(K,p)) +
  geom_col(aes(fill=vowel), position="dodge2") +
  facet_wrap(~experiment+new)

############################
# Old (pi_2) vs new (pi_3) #
############################

# Differences are (property value of new model) - (property value of old model)

cmp = final[experiment=="compare"]

setkey(cmp, "vowel", "K", "new")

groups = cmp[, .(group = runif(1)), key = .(vowel, K)] # Assign unique group labels
cmp = cmp[groups]

ggplot(cmp, aes(logz, H)) +
  geom_path(aes(group=group, color=K), arrow=arrow(length=unit(0.3,"cm")), size=1) +
  ggtitle("Differences in information (H) and evidence (log Z) going from pi_2 -> pi_3")

diff = cmp[, .(
  d_logz = diff(logz),
  d_H = diff(H)
), key = .(vowel, K)]

summary(diff)

summary(diff[K==5])
