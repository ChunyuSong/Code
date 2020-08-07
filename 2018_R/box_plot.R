library(ggplot2)
library(dplyr)
library(reshape2)
library(tibble)

P0907_01 <- read.csv("Z:\\Peng_Sun\\Mar_MSADEM\\Processed_18_0905\\P0907_01.csv")
P0907_03 <- read.csv("Z:\\Peng_Sun\\Mar_MSADEM\\Processed_18_0905\\P0907_03.csv")
P0907_07 <- read.csv("Z:\\Peng_Sun\\Mar_MSADEM\\Processed_18_0905\\P0907_07_new.csv")
P0907_21 <- read.csv("Z:\\Peng_Sun\\Mar_MSADEM\\Processed_18_0905\\P0907_21.csv")
tex <- as_labeller(c(`FA`="DTI FA",`ADC`="DTI ADC",`Axial`="DTI Axial",`Radial`="DTI Radial",
                     `Fiber.Ratio`="DBSI Fiber Fraction",`Fiber.FA`="DBSI Fiber FA",
                     `Fiber.Axial`="DBSI Fiber Axial",`Fiber.Radial`="DBSI Fiber Radial",
                     `Restricted.Ratio`="DBSI Restricted Fraction",
                     `Hindered.Ratio`="DBSI Hindered Fraction",
                     `Free`="DBSI Free Fraction",
                     `Non.Restricted.Ratio`="DBSI Non Restricted Fraction"))

rawdata <- rbind(P0907_01,P0907_03,P0907_07,P0907_21)
# rawdata$Non.Restricted.Ratio <- rawdata$Restricted.Ratio + rawdata$Hindered.Ratio
rawdata <- add_column(rawdata, Non.Restricted.Ratio=rawdata$Restricted.Ratio+rawdata$Hindered.Ratio, .before="Hindered.Ratio")

data <- filter(rawdata, ROI.ID==1, FA<1, Fiber.Ratio>0.4, Fiber.Axial<2.5, Fiber.Radial>0.1 & Fiber.Radial<0.5)

fildata <- melt(data, measure.vars = colnames(data)[which(colnames(data)=="FA"):which(colnames(data)=="Non.Restricted.Ratio")], 
                id.vars = c("Sub_ID","Scan_ID"), variable.name = "Param", value.name = "value", na.rm = TRUE)

ggplot(data=fildata)+
  geom_boxplot(mapping = aes(x=Scan_ID,y=value,fill=Sub_ID))+
  # stat_summary(fun.y=mean, geom="point", shape=20, size=10, color="red", fill="red") +
  # geom_jitter(mapping = aes(x=Scan_ID,y=value,fill=Sub_ID))+
  facet_wrap(~Param, ncol = 4,scales = "free", labeller = tex)+
  ylim(0, NA)+
  # theme(axis.text.x = element_text(angle = 90, hjust = 1))
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title = element_blank(),
        legend.position = c(1,0), # c(0,0) bottom left, c(1,1) top-right, c(1,0) bottom right
        legend.justification = c(1,0)
        )

