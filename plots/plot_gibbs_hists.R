library(ggplot2)
library(dplyr)

setwd("/data/bnpy/output/gibbs_traces")
all_gibbs_chains <- read.csv("all_gibbs_chains.csv")
colnames(all_gibbs_chains) <- c("Chain","Iteration","K","Layers")

# burn-in: remove iterations below 50
all_gibbs_chains_warmed <- all_gibbs_chains %>% filter(Iteration >= 50) 

# Plot the data as a histogram over components
comp_bars <- ggplot(all_gibbs_chains, aes(factor(K),fill=factor(Layers))) 
comp_bars <- comp_bars + geom_bar(position=position_dodge(), alpha=0.65, binwidth=1)
comp_bars <- comp_bars + xlab("number of active components")
comp_bars <- comp_bars + theme(legend.title = element_text(size = 15))
comp_bars <- comp_bars + theme(legend.text = element_text(size = 15))
comp_bars <- comp_bars + theme(axis.title.y = element_text(size = 15))
comp_bars <- comp_bars + theme(axis.text.y = element_text(size = 15))
comp_bars <- comp_bars + theme(axis.title.x = element_text(size = 15))
comp_bars <- comp_bars + theme(axis.text.x = element_text(size = 15))
comp_bars