library(tidyverse)
library(scales)

# AP: 63

data <- read.csv("results/wiki_alr.csv")

ylabel = "Spearman's rho"
# MEN: 0.65, 0.75
l = c(0.65, 0.75)
b = seq(0.65, 0.75, 0.025)
eval_name = "MEN"

# WS353
#l = c(0.6, 0.675)
#b = seq(0.6, 0.675, 0.025)
#eval_name = "WS353"

# RW
#l = c(0.25, 0.35)
#b = seq(0.25, 0.35, 0.025)
#eval_name = "RW"

# RG65
#l = c(0.725, 0.8)
#b = seq(0.725, 0.8, 0.025)
#eval_name = "RG65"

# Google
#l = c(0.275, 0.65)
#b = seq(0.275, 0.65, 0.05)
#eval_name = "Google"
#ylabel = "Accuracy"

# AP
#l = c(0.5, 0.65)
#b = seq(0.5, 0.65, 0.025)
#eval_name = "AP"
#ylabel = "Clusters Purity"

# Battig
l = c(0.4, 0.48)
b = seq(0.4, 0.48, 0.025)
#eval_name = "Battig"
#ylabel = "Clusters Purity"

# SemEval
l = c(0.17, 0.19)
b = seq(0.17, 0.19, 0.0125)
eval_name = "SemEval2012_2"
ylabel = "Clusters Purity"

data[c(1,2,3,4,5,6,7), ] %>% ggplot(aes(x=Sampling.Rate, y=SemEval2012_2)) + 
    geom_bar(stat="identity", fill="tomato3") + 
    scale_y_continuous(breaks=b, limits=l, oob = rescale_none) + 
    theme(
        axis.text=element_text(size=22),
        axis.title=element_text(size=22, face="bold"),
        plot.title=element_text(size=26, face="bold", hjust=0.5),
        legend.text=element_text(size=20),
        legend.title=element_text(size=22),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x="Sampling Rate", y=ylabel) +
    geom_hline(yintercept=data[c(9), eval_name]) +
    annotate("text", 0.01, data[c(9), eval_name], hjust= -0.1, vjust = -1, label = "Baseline", size=8) +
    ggtitle(eval_name)

ggsave(paste(eval_name,".png",sep=""))
ggsave(paste(eval_name,".eps",sep=""))