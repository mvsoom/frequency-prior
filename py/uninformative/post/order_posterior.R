##############################################
# Posterior values of the model order (P, Q) #
##############################################
source("run_stats.R")

rs = full[, .(new, P, Q,
              best = logz == max(logz),
              logz,
              joint_prob = normalize(exp(-(max(logz) - logz)))),
          by=.(file)]

# MAP per file
# All vowels are clearly Q = 5
rs[best==T]

# Top 3 per file in percent
# Only "you" is 100% sure P = 10
# We need averaging over P
rs[order(-joint_prob), .SD[1:3, .(best, P, Q, rel_prob = pct(joint_prob))], key=.(file)]

rs[, sum(joint_prob), key = P][, .(prob = pct(normalize(V1)))]

# Best Q values for new=False models? Between 2 and 4.
rs_false = full[new==F, .(new, P, Q,
                          best = logz == max(logz),
                          logz,
                          joint_prob = normalize(exp(-(max(logz) - logz)))),
                by=.(file)]

rs_false[best==T]

# Top 3
rs_false[order(-joint_prob), .SD[1:3, .(best, P, Q, rel_prob = pct(joint_prob))], key=.(file)]