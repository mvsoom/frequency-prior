setwd("~/WRK/proj/formant-prior/research/data")

library(data.table)
library(ggplot2)
library(ggExtra)
theme_set(theme_bw())
theme_update(panel.grid.minor.x = element_blank())

d = as.data.table(read.csv(file = "pb.csv",header = T))

setnames(d, c("F1", "F2", "F3"), c("x", "y", "z"))

d[, main := substr(Vowel, 1, 1)]
d = d[, .(Type, Sex, Speaker, Vowel, main, IPA, F0, x, y, z)]
summary(d)

c = min(d$x)
d[, `:=`(r = log(x/c), s = log(y/x), t = log(z/y))]

ggMarginal(
  ggplot(d, aes(r, s, color=main)) +
    geom_vline(xintercept = mean(d$r)) +
    geom_hline(yintercept = mean(d$s)) +
    geom_point() +
    geom_density2d()
  , type="histogram"
)
