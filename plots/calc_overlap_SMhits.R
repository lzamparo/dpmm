library(ggplot2)
library(dplyr)

# load SM hit list, grab hits
setwd('/data/RAD52/results/ranked_hits/')
sm_hits <- read.csv('SM_RankedHits_NoDubs.csv')
sm_hits <- sm_hits %>% filter(Score >= 40)

# load scored lists for 3 layer models
setwd("/data/bnpy/scored_lists/")
three_layer_0.5 <- read.csv("bnpy_3layer_gamma_0.5_df.csv")
cols <- colnames(three_layer_0.5)
three_layer_0.5 <- cbind(three_layer_0.5,factor(0.5))
colnames(three_layer_0.5)[8] <- "alpha"

three_layer_1.0 <- read.csv("bnpy_3layer_gamma_1.0_df.csv")
cols <- colnames(three_layer_1.0)
three_layer_1.0 <- cbind(three_layer_1.0,factor(1.0))
colnames(three_layer_1.0)[8] <- "alpha"

three_layer_2.0 <- read.csv("bnpy_3layer_gamma_2.0_df.csv")
cols <- colnames(three_layer_2.0)
three_layer_2.0 <- cbind(three_layer_2.0,factor(2.0))
colnames(three_layer_2.0)[8] <- "alpha"

three_layer_3.0 <- read.csv("bnpy_3layer_gamma_3.0_df.csv")
cols <- colnames(three_layer_3.0)
three_layer_3.0 <- cbind(three_layer_3.0,factor(3.0))
colnames(three_layer_3.0)[8] <- "alpha"

three_layer_ctrl <- read.csv("control_3layer_df.csv")
cols <- colnames(three_layer_ctrl)
three_layer_ctrl <- cbind(three_layer_ctrl,factor('ctrl'))
colnames(three_layer_ctrl)[8] <- "alpha"

# Compute overlap with sm orf hits: top 10, top 20, top 50 by overlap w sm hits
sm_ORFs <- sm_hits$ORF
three_df_names <- c("three_layer_0.5","three_layer_1.0","three_layer_2.0","three_layer_3.0","three_layer_ctrl")
three_dfs <- c(three_layer_0.5,three_layer_1.0,three_layer_2.0,three_layer_3.0,three_layer_ctrl)

for (i in 1:length(three_dfs)) {
  top_10 <- three_dfs[i] %>% arrange(desc(Score)) %>% slice(1:10) 
  top_20 <- three_dfs[i] %>% arrange(desc(Score)) %>% slice(1:20)
  top_50 <- three_dfs[i] %>% arrange(desc(Score)) %>% slice(1:50)
  overlap_10 <- sum(top_10$ORF %in% sm_hits$ORF)
  overlap_20 <- sum(top_10$ORF %in% sm_hits$ORF)
  overlap_50 <- sum(top_10$ORF %in% sm_hits$ORF) 
}

