setwd("~/WRK/proj/formant/")

library(data.table)
library(ggplot2)
library(ggExtra)
theme_set(theme_bw())
theme_update(panel.grid.minor.x = element_blank())

pb = as.data.table(read.csv(file = "pb.csv",header = T))
h = as.data.table(read.csv(file = "hillenbrand-steady-state.csv",header = T))

pb[, dataset := "PB"]
h[, dataset := "H"]

# Derive Sex from Type
h$Sex = h$Type
levels(h$Sex) <- list(m = c("b", "m"), f = c("g", "w"))

# Map "boy" and "girl" into "chld"
levels(h$Type) <- list(m = "m", w = "w", c = c("b", "g"))

# Bind
common = c("dataset", "Sex", "Type", "Vowel", "F0", "F1", "F2", "F3") # intersect(names(pb), names(h))

pbh = rbind(pb[, ..common], h[, ..common])

# Remove any rows with NAs
pbh = pbh[!(is.na(F1) | is.na(F2) | is.na(F3))]

# Mark vowels that were present in both experiments as `shared_vowel = 1`, otherwise a 0
table(pbh$dataset, pbh$Vowel)

sv = pbh[, .(shared_vowel = as.numeric(("PB" %in% dataset) & ("H" %in% dataset))), by = .(Vowel)]

pbh = pbh[sv, on = "Vowel"]
pbh = pbh[, .(dataset, Sex, Type, Vowel, shared_vowel, F0, F1, F2, F3)]

# Output
write.csv(pbh, "pbh-steady-state.csv", quote=F, row.names = F)
pbh
