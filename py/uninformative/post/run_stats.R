setwd("~/WRK/proj/formant-prior/research/py/uninformative/post/")

library(data.table)
library(ggplot2)
library(ggExtra)
theme_set(theme_bw())
theme_update(panel.grid.minor.x = element_blank())

pct = function(x) round(x*100)
pct_text = function(x) sprintf("%d﹪", pct(x))
normalize = function(x) x/sum(x)

# The phoneme is always the first vowel in the example word, e.g.
# shOre instead of shorE for the first element.
vowels = list(
  # ɔ in /ʃɔː(r)/ ("shore")
  # awb/arctic_a0094
  # F0 = 113 Hz
  `shore` = "awb/arctic_a0094",
  # æ in /ðæt/ ("that")
  # bdl/arctic_a0017
  # F0 = 138 Hz
  `that` = "bdl/arctic_a0017",
  # u in /juː/ ("you")
  # jmk/arctic_a0067
  # F0 = 96
  `you` = "jmk/arctic_a0067",
  # ɪ in /ˈlɪt(ə)l/ ("little")
  # rms/arctic_a0382
  # F0 = 102 Hz
  `little` = "rms/arctic_a0382",
  # ə in /ənˈtɪl/ ("until")
  # slt/arctic_b0041
  # F0 = 110 Hz
  `until` = "slt/arctic_b0041"
)

experiments = list(
  `compare` = "hyper_cmp",
  `free` = "hyper_free",
  `free12` = "hyper_free12"
)

full = as.data.table(read.csv(file = "run_stats.csv",header = T))
full[, vowel := file]
levels(full$vowel) <- vowels
full$hyperfile = as.factor(full$hyperfile)
full[, experiment := hyperfile]
levels(full$experiment) <- experiments
full$experiment = ordered(full$experiment)
full[, `:=`(new = new == "True")]
full$P = ordered(full$P)
full$Q = ordered(full$Q)

full_cmp = full[experiment == "compare"]
full_free = full[experiment == "free"]
full_free12 = full[experiment == "free12"]

# p(new,P,Q|hyper,vowel)
full_posterior = full[, .(new, P, Q,
                          MAP = logz == max(logz),
                          p = normalize(exp(-(max(logz) - logz)))),
                      by=.(experiment,vowel)]

full_posterior_cmp = full_posterior[experiment == "compare"]
full_posterior_free = full_posterior[experiment == "free"]
full_posterior_free12 = full_posterior[experiment == "free12"]