setwd("~/WRK/proj/formant-prior/research/py/uninformative/post/")

library(data.table)
library(ggplot2)
library(ggExtra)
theme_set(theme_bw())
theme_update(panel.grid.minor.x = element_blank())

pct = function(x) round(x*100)
normalize = function(x) x/sum(x)

# ɔ in /ʃɔː(r)/ ("shore")
# F0 = 113 Hz
# 05Feb21 13:07:30        input_file=bdl/arctic_a0017     output_file=post/run_stats.csv  Processing
# æ in /ðæt/ ("that")
# F0 = 138 Hz
# 05Feb21 13:07:30        input_file=jmk/arctic_a0067     output_file=post/run_stats.csv  Processing
# u in /juː/ ("you")
# F0 = 96
# 05Feb21 13:07:30        input_file=rms/arctic_a0382     output_file=post/run_stats.csv  Processing
# ɪ in /ˈlɪt(ə)l/ ("little")
# F0 = 102 Hz
# 05Feb21 13:07:30        input_file=slt/arctic_b0041     output_file=post/run_stats.csv  Processing
# ə in /ənˈtɪl/ ("until")
# F0 = 110 Hz
full = as.data.table(read.csv(file = "run_stats.csv",header = T))
full[, `:=`(new = new == "True")]

# Occam factors
# Not so thrustworthy because (log L)_max (i.e. logl) values are not
# close enough to each other for a faithful decomposition of log Z into
# (log L)_max and log W (i.e. Occam factor). Better to just compare
# the log Z values as their errors should be smaller due to the
# averaging.
occ = full[, .(logw = logz - logl, logl, logz), key=.(file, P, Q, new)]

docc = occ[, .(
  delta_logw = .SD[new==T, logw] - .SD[new==F, logw],
  delta_logl = .SD[new==T, logl] - .SD[new==F, logl],
  delta_logz = .SD[new==T, logz] - .SD[new==F, logz]
  ), key = .(file, P, Q)]

ggplot(occ[Q==5], aes(y=logw, color=new, shape=new)) +
  geom_point(aes(P)) +
  facet_wrap(~file)

ggplot(docc[Q==5], aes(y=delta_logw)) +
  geom_point(aes(P)) +
  facet_wrap(~file)

ggplot(docc[Q==5], aes(y=delta_logl)) +
  geom_point(aes(P)) +
  facet_wrap(~file)

ggplot(docc[Q==5], aes(y=delta_logz)) +
  geom_point(aes(P)) +
  facet_wrap(~file)

###


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
