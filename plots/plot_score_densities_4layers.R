library(ggplot2)
library(dplyr)

# first plots to try:  highest ELBO 4-layer vs GMM control. 

setwd("/data/bnpy/scored_lists/")

four_layer_0.5 <- read.csv("bnpy_4layer_gamma_0.5_df.csv")
cols <- colnames(four_layer_0.5)
four_layer_0.5 <- cbind(four_layer_0.5,factor(0.5))
colnames(four_layer_0.5)[8] <- "alpha"

four_layer_1.0 <- read.csv("bnpy_4layer_gamma_1.0_df.csv")
cols <- colnames(four_layer_1.0)
four_layer_1.0 <- cbind(four_layer_1.0,factor(1.0))
colnames(four_layer_1.0)[8] <- "alpha"

four_layer_2.0 <- read.csv("bnpy_4layer_gamma_2.0_df.csv")
cols <- colnames(four_layer_2.0)
four_layer_2.0 <- cbind(four_layer_2.0,factor(2.0))
colnames(four_layer_2.0)[8] <- "alpha"

four_layer_3.0 <- read.csv("bnpy_4layer_gamma_3.0_df.csv")
cols <- colnames(four_layer_3.0)
four_layer_3.0 <- cbind(four_layer_3.0,factor(3.0))
colnames(four_layer_3.0)[8] <- "alpha"

four_layer_ctrl <- read.csv("control_4layer_df.csv")
cols <- colnames(four_layer_ctrl)
four_layer_ctrl <- cbind(four_layer_ctrl,factor('ctrl'))
colnames(four_layer_ctrl)[8] <- "alpha"

# join 3-layer models GMM control
four_layers <- rbind(four_layer_0.5,four_layer_1.0,four_layer_2.0,four_layer_3.0,four_layer_ctrl)

# No cells less than 50
four_layers <- four_layers %>% filter(Cells >= 50)
big_scores <- four_layers %>% filter(Score >= 0.75)

# 4 layer histogram
four_layer_histogram <- ggplot(four_layers,aes(x=Score, fill=alpha)) + geom_histogram(binwidth=.3, position="dodge")
four_layer_histogram <- four_layer_histogram + theme(legend.title = element_text(size = 17))
four_layer_histogram <- four_layer_histogram + theme(legend.text = element_text(size = 17))
four_layer_histogram <- four_layer_histogram + theme(axis.title.y = element_text(size = 17))
four_layer_histogram <- four_layer_histogram + theme(axis.text.y = element_text(size = 17))
four_layer_histogram <- four_layer_histogram + theme(axis.text.x = element_text(size = 17))
four_layer_histogram <- four_layer_histogram + theme(axis.title.x = element_text(size = 17))
four_layer_histogram 
