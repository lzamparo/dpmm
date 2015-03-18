library(ggplot2)
library(dplyr)

setwd("/data/bnpy/output/gibbs_traces")
all_gibbs_chains <- read.csv("all_gibbs_chains.csv")
colnames(all_gibbs_chains) <- c("Chain","Iteration","K","Layers")

# burn-in: remove iterations below 50
#all_gibbs_chains_warmed <- all_gibbs_chains %>% filter(Iteration >= 50) 

# Re-label the facets
facet_names <- list('3'="3 layer SdA",'4'="4 layer SdA")
facet_labeller <- function(variable,value){
    return(facet_names[as.character(value)])
}

# Plot the traces over K as line plots, faceting by Layer 
trace_plot <- ggplot(all_gibbs_chains, aes(x=Iteration, y=K, colour=factor(Chain))) 
trace_plot <- trace_plot + geom_line(alpha=0.65)
trace_plot <- trace_plot + geom_vline(xintercept = 50, colour="black", linetype = "longdash")
trace_plot <- trace_plot + xlab("iterations")
trace_plot <- trace_plot + ylab("active components")
trace_plot <- trace_plot + labs(colour = "Chains")
trace_plot <- trace_plot + facet_grid(Layers ~ ., labeller=facet_labeller)
trace_plot <- trace_plot + theme(legend.title = element_text(size = 15))
trace_plot <- trace_plot + theme(legend.text = element_text(size = 15))
trace_plot <- trace_plot + theme(strip.text.y = element_text(size = 15))
trace_plot <- trace_plot + theme(axis.title.y = element_text(size = 15))
trace_plot <- trace_plot + theme(axis.text.y = element_text(size = 15))
trace_plot <- trace_plot + theme(axis.title.x = element_text(size = 15))
trace_plot <- trace_plot + theme(axis.text.x = element_text(size = 15))
trace_plot

# Plot the traces over alpha