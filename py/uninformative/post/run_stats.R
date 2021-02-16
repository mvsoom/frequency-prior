setwd("~/WRK/proj/formant-prior/research/py/uninformative/post/")

library(data.table)
library(ggplot2)
library(ggExtra)
theme_set(theme_bw())
theme_update(panel.grid.minor.x = element_blank())

pct = function(x) round(x*100)
normalize = function(x) x/sum(x)

# ɔ in /ʃɔː(r)/("shore")
# bdl/arctic_a0017
# æ in /ðæt/ ("that")
# F0 = 138 Hz

# u in /juː/ ("you")
# jmk/arctic_a0067
# F0 = 96

# ɪ in /ˈlɪt(ə)l/ ("little")
# rms/arctic_a0382
# F0 = 102 Hz

# ə in /ənˈtɪl/ ("until")
# slt/arctic_b0041
# F0 = 110 Hz
full = as.data.table(read.csv(file = "run_stats.csv",header = T))
full[, `:=`(new = new == "True")]