library(tidyverse)

data <- read.csv("data_to_runtime.csv")
t <- data[c(5,6,7,8),] %>% ggplot(aes(x = Percentage, y = Minutes, fill = "tomato3")) +
    geom_bar(stat = "identity") +
    scale_x_discrete(limits=c("25%", "50%", "75%", "100%")) +
    theme(axis.text=element_text(size=22),
        axis.title=element_text(size=22, face="bold"),
        plot.title=element_text(size=24, face="bold", hjust=0.5),
        legend.text=element_text(size=20),
        legend.title=element_text(size=22),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_y_continuous(breaks=seq(0,150,25)) +
    ggtitle("Wikipedia") +
    guides(fill=FALSE)

ggsave("data_time_scaling_wiki.pdf", t)
ggsave("data_time_scaling_wiki.png", t)

t <- data[c(1,2,3,4),] %>% ggplot(aes(x = Percentage, y = Minutes, fill = "tomato3")) +
    geom_bar(stat = "identity") +
    scale_x_discrete(limits=c("25%", "50%", "75%", "100%")) +
    theme(axis.text=element_text(size=22),
        axis.title=element_text(size=22, face="bold"),
        plot.title=element_text(size=24, face="bold", hjust=0.5),
        legend.text=element_text(size=20),
        legend.title=element_text(size=22),
        axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_y_continuous(breaks=seq(0,700,100)) +
    ggtitle("Web") +
    guides(fill=FALSE)

ggsave("data_time_scaling_web.pdf", t)
ggsave("data_time_scaling_web.png", t)