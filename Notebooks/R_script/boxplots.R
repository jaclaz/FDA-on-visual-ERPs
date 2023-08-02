library( car ) 
library( ellipse )
library( faraway ) 
library( leaps )
library( MASS )
library( GGally)
library( rgl )
library(RColorBrewer)
library(ggplot2)

pluto=read.table("scores.csv",sep=",",header=TRUE)
attach(pluto)

pluto$Category=factor(pluto$Category, labels=c('Animal', 'Body Part', 'Vehicle', 'Tool', 'Food'))
pluto$X.1=factor(pluto$X.1, labels=c('1','2','3','4'))

G <- ggplot(data = pluto,
            aes(x = X.1, y = X0))
x11()
G +  scale_fill_brewer(palette = 'Set2') +
  geom_boxplot(aes(fill = Category), show.legend = T) +
  #facet_wrap(~struc, nrow = 3) +
  theme_minimal(base_size =  20) +
  theme(legend.position = 'bottom') +
  labs(x = 'Principal Component Function', y = 'Score') +
  ggtitle('Scores of the Functional Components')

ggsave('scores_fpca.png', plot = last_plot(), dpi = 300)
