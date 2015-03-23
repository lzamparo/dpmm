# script to load 3-layer dfs

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