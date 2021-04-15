source("run_stats.R")

# To be complete we should the errorbars for lrf, but std(lrf) is only O(.5)
top2 = full[experiment %in% c("compare","free"), .(lrf = log(factorial(as.integer(Q))), ldiff = diff(sort(logz, decreasing = T))[[1]]), key=.(Q,experiment,vowel,new)]

ggplot(top2
       , aes(abs(ldiff), lrf, color=experiment, shape=new)) +
  geom_jitter(size=2, height=0.1, width = 0.1) +
  geom_abline(slope = 1) +
  coord_equal() +
  ylab("redundancy factor log Q!") +
  xlab("|log Z (best model) - log Z (second best model)|") +
  ggtitle("Accounting for redundancy matters for model comparison in our problem")

write.csv(top2[, .(ldiff, lrf)], 'evidence_diff.csv', quote=F, row.names=F)