##############################################
# Posterior values of the model order (P, Q) #
##############################################
source("run_stats.R")

# Plot p(P, Q | new,vowel)
posterior_PQ = full_posterior_cmp[, .(P, Q, prob=normalize(p)), key=.(vowel,new)]

ggplot(
  posterior_PQ
) +
  geom_tile(aes(P, Q, fill=prob)) +
  geom_text(aes(P, Q, label=pct(prob)), size=2, color="white") +
  coord_equal() +
  scale_fill_gradient2() +
  facet_wrap(~new+vowel, nrow=2) +
  ggtitle("p(P, Q | new, vowel) shows different patterns of the model order for the old (FALSE) and new (TRUE) priors")

ggsave("model_order_posterior_cmp.png")

# Check top 3, which holds 99% of posterior mass given vowel
# Top 3 is always (new=T,Q=5) models
top3 = full_posterior_cmp[order(-p), .SD[1:3, .(MAP, new, P, Q, p)], key=.(vowel)]
top3[, `prob (%)` := pct(p)]

top3[, sum(`prob (%)`), key=vowel]
top3

# Does the data prefer the old or new prior?
# Answer: the new prior, by 100% to over 20 decimal places
posterior_given_vowel = function(...) full_posterior_cmp[, .(prob = sum(p)), key=c("vowel", c(...))]

posterior_given_vowel("new")

ggplot(
  posterior_given_vowel("P")
  ) +
  geom_col(aes(P, normalize(prob))) +
  ggtitle("Posterior probability of the trend order")

ggsave("trend_order_posterior_cmp.png")

# Find the MAP in (new, Q) dimensions
ggplot(posterior_given_vowel("new", "Q")) +
  geom_text(aes(new, Q, label=pct_text(prob))) +
  facet_wrap(~vowel) +
  ggtitle("p(new, Q | vowel) shows very accurate MAP approximation by (new=True,Q=5)")

MAP_posterior = full_posterior_cmp[new==T & Q==5]
MAP_posterior[, p := normalize(p), key=.(vowel,new,Q)]

ggplot(
  MAP_posterior
) +
  geom_col(aes(P, p)) +
  facet_wrap(~vowel) +
  ggtitle("p(P | new=True, Q=5, vowel)")