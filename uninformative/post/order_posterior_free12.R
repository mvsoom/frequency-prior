##############################################
# Posterior values of the model order (P, Q) #
##############################################
source("run_stats.R")

# Plot p(P, Q | new,vowel)
posterior_PQ = full_posterior_free12[, .(P, Q, prob=normalize(p)), key=.(vowel)]

ggplot(
  posterior_PQ
) +
  geom_tile(aes(P, Q, fill=prob)) +
  geom_text(aes(P, Q, label=pct(prob)), size=2, color="black") +
  coord_equal() +
  scale_fill_gradient2() +
  facet_wrap(~vowel, nrow=2) +
  ggtitle("p(P, Q | new, vowel)")

ggsave("model_order_posterior_free12.png")

# Check top 2, which holds 99% of posterior mass given vowel
top2 = full_posterior_free12[order(-p), .SD[1:2, .(MAP, P, Q, p)], key=.(vowel)]
top2[, `prob (%)` := pct(p)]

top2[, sum(`prob (%)`), key=vowel]
top2

posterior_given_vowel = function(...) full_posterior_free12[, .(prob = sum(p)), key=c("vowel", c(...))]

ggplot(
  posterior_given_vowel("P")
) +
  geom_col(aes(P, normalize(prob))) +
  ggtitle("Posterior probability of the trend order")

ggsave("trend_order_posterior_free12.png")

ggplot(
  posterior_given_vowel("Q")
) +
  geom_col(aes(Q, normalize(prob))) +
  ggtitle("Posterior probability of the cyclical order")
