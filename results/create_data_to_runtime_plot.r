library(tidyverse)

data <- read.csv("data_to_runtime.csv")
t <- data %>% gather(Time, Minutes, Reduce:Merging) %>% ggplot(aes(x = Percentage, y = Minutes, fill = Time)) + geom_bar(stat = "identity") + scale_x_discrete(limits=c("web 25%", "web 50%", "web 75%", "web 100%", "wiki 25%", "wiki 50%", "wiki 75%", "wiki 100%")) +  theme(axis.text=element_text(size=16), axis.title=element_text(size=16, face="bold"), plot.title=element_text(size=18, face="bold", hjust=0.5), legend.text=element_text(size=14), legend.title=element_text(size=16), axis.text.x = element_text(angle = 45, hjust = 1)) + scale_y_continuous(breaks=seq(0,500,50))

ggsave("data_time_scaling.pdf", t)
