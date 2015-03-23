library(ggplot2)
library(dplyr)

computeOverlap <- function(df,sm_hits){
  top_10 <- df %>% arrange(desc(Score)) %>% slice(1:10) 
  top_20 <- df %>% arrange(desc(Score)) %>% slice(1:20)
  top_50 <- df %>% arrange(desc(Score)) %>% slice(1:50)
  overlap_10 <- sum(top_10$ORF %in% sm_hits$ORF)
  overlap_20 <- sum(top_20$ORF %in% sm_hits$ORF)
  overlap_50 <- sum(top_50$ORF %in% sm_hits$ORF) 
  c(overlap_10,overlap_20,overlap_50)
}

# load SM hit list, grab hits
setwd('/data/RAD52/results/ranked_hits/')
sm_hits <- read.csv('SM_RankedHits_NoDubs.csv')
sm_hits <- sm_hits %>% filter(Score >= 40)

# load scored lists for 3 layer models
setwd('~/projects/dpmm/plots')
source('load_3layer_dfs.R')
three_df_names <- c("three_layer_0.5","three_layer_1.0","three_layer_2.0","three_layer_3.0","three_layer_ctrl")

# Compute overlap with sm orf hits: top 10, top 20, top 50 by overlap w sm hits
scores <- data.frame(rbind(computeOverlap(three_layer_0.5,sm_hits),computeOverlap(three_layer_1.0,sm_hits),computeOverlap(three_layer_2.0,sm_hits),computeOverlap(three_layer_3.0,sm_hits),computeOverlap(three_layer_ctrl,sm_hits)))

# load scored lists for 4 layer models
setwd('~/projects/dpmm/plots')
source('load_4layer_dfs.R')
four_df_names <- c("four_layer_0.5","four_layer_1.0","four_layer_2.0","four_layer_3.0","four_layer_ctrl")

# Compute overlap with sm orf hits: top 10, top 20, top 50 by overlap w sm hits
scores <- data.frame(rbind(scores,computeOverlap(four_layer_0.5,sm_hits),computeOverlap(four_layer_1.0,sm_hits),computeOverlap(four_layer_2.0,sm_hits),computeOverlap(four_layer_3.0,sm_hits),computeOverlap(four_layer_ctrl,sm_hits)))

names <- c(three_df_names,four_df_names)
annotated_scores <- cbind(scores,names)
colnames(annotated_scores) <- c("top 10","top 20", "top 50", "model")

setwd("/data/bnpy/scored_lists/")
write.csv(annotated_scores,"overlap_table.csv")



