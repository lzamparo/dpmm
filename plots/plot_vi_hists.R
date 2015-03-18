library(ggplot2)
library(dplyr)

setwd("/data/bnpy/output/vi_processed_output/")
all_vi_tasks <- read.csv("all_vi_component_traces.csv")
colnames(all_vi_tasks) <- c("Iteration","K","Layers")
all_vi_tasks$Layers <- as.factor(all_vi_tasks$Layers)

vi_comp_bars <- ggplot(all_vi_tasks, aes(factor(K),fill=Layers))
vi_comp_bars <- vi_comp_bars + geom_bar(position=position_dodge(), alpha=0.65, binwidth=1)
vi_comp_bars <- vi_comp_bars + xlab("number of active components")
vi_comp_bars <- vi_comp_bars + theme(legend.title = element_text(size = 15))
vi_comp_bars <- vi_comp_bars + theme(legend.text = element_text(size = 15))
vi_comp_bars <- vi_comp_bars + theme(axis.title.y = element_text(size = 15))
vi_comp_bars <- vi_comp_bars + theme(axis.text.y = element_text(size = 15))
vi_comp_bars <- vi_comp_bars + theme(axis.title.x = element_text(size = 15))
vi_comp_bars <- vi_comp_bars + theme(axis.text.x = element_text(size = 15))
vi_comp_bars