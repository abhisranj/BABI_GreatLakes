##
## Kaggle Competition - Ultrasound Nerve Segmentation
## Use bioconductor(R) package to overlay mask with mri image
##

#Load libraries
##Probably will need to download bioconductor package
##source("http://bioconductor.org/biocLite.R")
##biocLite("EBImage")
library("EBImage")

#Read file
input_dir = "../input/train/"
files = dir("../input/train/")
imgs = files[seq(1, 100, 2)] #Limit to first 100 for memory reasons
masks = files[seq(2, 100, 2)] #Limit to first 100 for memory reasons

#Get back all image entries by metaprogramming
get_images <- function() {
  for(i in 1:length(imgs)) {
    fName <- paste("imgs.", i, sep="")
    assign(fName, eval(
      substitute(
        toRGB(readImage(paste0(input_dir, imgs[i]))) #Read and convert to rgb
      )
    ),
    envir=parent.frame()
    )
  }
}
get_images()

#Get back all masks entries by metaprogramming
get_masks <- function() {
  for(i in 1:length(imgs)) {
    fName <- paste("imgs_masks.", i, sep="")
    assign(fName, eval(
      substitute(
        readImage(paste0(input_dir, masks[i]))
      )
    ),
    envir=parent.frame()
    )
  }
}
get_masks()

#Display overlaid mri based on number
display_mri = function(entry){
  #Prep data
  mask = get(paste0("imgs_masks.", entry))
  img = get(paste0("imgs.", entry))
  
  #Paint overlay and display
  overlaid_mri = paintObjects(mask , img, col=c("red"))
  display(overlaid_mri, method="raster")
}