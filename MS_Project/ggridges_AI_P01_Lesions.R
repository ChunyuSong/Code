library(dplyr)
library(ggpubr)
library(ggplot2)
library(ggsignif)
library(lemon)
library(grid)
library(gridExtra)
library(export)
library(tidyr)
library(janitor)
library(magrittr)
library(readr)
library(tidyverse)
library(ggforce)
library(pipeR)
library(officer)
library(rvg)
library(flextable)
library(ggridges)
library(ggsci)


g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

rm(list=ls())
max_limits <- c(1,2,3,2,1,1.3,3,1.3,0.5,1,0.5,4,1.5,10,4,1.5,0.5)
min_limits <- c(0,0,0,0,0,0,0,0,NA,NA,NA,NA,NA,NA,NA,NA,NA)
ncol = 4
nrow = 3
width = 2
height = 2
left <- rep(seq.int(from=1,by=width,length.out = ncol),nrow)
top <- sort(rep(seq.int(from=1,by=height,length.out = nrow),ncol))
file = 'Z:/Peng_Sun/AI/P01_lesions/processing/merged_table_complete_20190312.csv'
groups = 'ROI_CLASS'
leg_t = 0.5
leg_l = sum(unique(left))/ncol
grp_levels = c('1','2','4','5','6')
replace_levels <- c('NAWM','PBH','PGH','AGH','T2W')
recode_levels <- c('1','2','3','4','5')

doc <<- read_pptx()

plot_proc <- function(x){
  data <- filter(df_parameters[[x]])
  p1 <- ggplot(data, aes(y = data[[groups]])) + 
        geom_density_ridges(aes(x = VALUE, 
                                fill=data[[groups]]),alpha=0.7) +
        scale_fill_hue(c=85, l=40) +
        # scale_fill_manual(values = c("#66D871",  "#FFA59E", "#00AFBB", "#E7B800", "#FC4E07")) +
        scale_fill_lancet() +
        # scale_fill_brewer(palette = 'rainbow') +
        theme_minimal() + theme(legend.position='bottom') + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),legend.title = element_blank())  %T>% {as.character(unique(data[[groups]])) ->> group} +
        facet_wrap(~PARAMETER, ncol=3,scales='free') +
        theme(strip.text.x = element_text(size = 8, face="bold")) +
        labs_pubr() +  
        # theme(axis.title.y=element_blank(),
        #        axis.text.y=element_blank(),
        #        axis.ticks.y=element_blank()) +
               {if (exists("max_limits")) scale_x_continuous(breaks = scales::pretty_breaks(3), limits = c(min_limits[x],max_limits[x]),labels = scales::number_format(accuracy=0.1)) else scale_x_continuous(breaks = scales::pretty_breaks(3),labels = scales::number_format(accuracy=0.1))}  +
         rremove("legend.title")  + 
         rremove("xylab")
  if (x==1 || x==ncol*nrow+1) {
    counter <<- 1
    leg <- g_legend(p1)
    # Convert to a ggplot and print
    leg_plot <- as_ggplot(leg)
    doc <<- add_slide(doc, 'Title and Content', 'Office Theme')
    doc <<- ph_with_vg_at(doc, ggobj = leg_plot,
                          height = 0.5, width = 3, left = leg_l, top = leg_t)
  }
  p1 <- p1 + guides(fill=FALSE)
  doc <<- ph_with_vg_at(doc, ggobj = p1,
                        height = height, width = width, left = left[counter], top = top[counter])
  counter <<- counter + 1
}

run <- function(doc){
  mydata <<- read_csv(file) %>%
    clean_names(case='all_caps') %>% 
    mutate(!!groups:=factor(.[[groups]], levels=grp_levels, labels=recode_levels)) %>% 
    rename_all(funs(str_replace(., "_HDR", ""))) %>% 
    rename_all(funs(str_replace(., "_MAP", ""))) %>%
    rename_all(funs(str_replace(., "_MNI152", ""))) %>%
    rename_all(funs(str_replace(., "FIBER1", "FIBER"))) %>%
    select(!!groups,DTI_FA:WATER_RATIO,ISO_ADC,T2W_CSF_NORMALIZED:B0_CSF_NORMALIZED,T1W_DIFFRATIO,MTC) %>%
    filter(!is.na(.[[groups]])) %>% 
    # mutate(!!groups:= factor(.[[groups]], levels = grp_levels)) %>%
    gather(key='PARAMETER',value='VALUE',DTI_FA:MTC,factor_key = TRUE) %T>% 
    # mutate(PARAMETER = factor(PARAMETER, levels=c("DTI.FA","DTI.Axial","DTI.Radial","DBSI.Fiber.Fraction","DBSI.Fiber.Axial","DBSI.Fiber.Radial","DBSI.Fiber.Intra.Fraction","DBSI.Fiber.Intra.Axial","DBSI.Fiber.Intra.Radial"))) %T>%
    # mutate(PARAMETER = factor(PARAMETER, levels=c("DBSI.Restricted.Fraction","DBSI.NonRestricted.Fraction","DBSI.IA.Restricted.Fraction","DBSI.IA.NonRestricted.Fraction"))) %>%
    group_by_at(vars(!!groups,PARAMETER,VALUE)) %T>%
    {split(.,.$PARAMETER) ->> df_parameters}
  lapply(1:(length(df_parameters)), FUN=function(x) plot_proc(x))
  print(doc, target = "Z:/Peng_Sun/AI/P01_lesions/processing/ggridges_lancet_with_ticks.pptx")
}

run(doc)
