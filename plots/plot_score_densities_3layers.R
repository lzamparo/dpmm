library(ggplot2)
library(dplyr)

# first plots to try: highest ELBO 3-layer vs GMM control 

setwd("/data/bnpy/scored_lists/")
#3_layer_score_files <- c("bnpy_3layer_gamma_0.5_df.csv", "bnpy_3layer_gamma_2.0_df.csv", "bnpy_3layer_gamma_1.0_df.csv", "bnpy_3layer_gamma_3.0_df.csv")

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

# join 3-layer models GMM control
three_layers <- rbind(three_layer_0.5,three_layer_1.0,three_layer_2.0,three_layer_3.0,three_layer_ctrl)

# No cells less than 50
three_layers <- three_layers %>% filter(Cells >= 50)
big_scores <- three_layers %>% filter(Score >= 0.75)

# histogram plot
three_layer_histogram <- ggplot(three_layers,aes(x=Score, fill=alpha)) + geom_histogram(binwidth=.3, position="dodge")
three_layer_histogram <- three_layer_histogram + theme(legend.title = element_text(size = 17))
three_layer_histogram <- three_layer_histogram + theme(legend.text = element_text(size = 17))
three_layer_histogram <- three_layer_histogram + theme(axis.title.y = element_text(size = 17))
three_layer_histogram <- three_layer_histogram + theme(axis.text.y = element_text(size = 17))
three_layer_histogram <- three_layer_histogram + theme(axis.text.x = element_text(size = 17))
three_layer_histogram <- three_layer_histogram + theme(axis.title.x = element_text(size = 17))
three_layer_histogram 