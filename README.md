# Wine-Quality-Classification

**Project Description**  
Vinho verde is a medium-alcohol wine from the Minho (northwest) region of Portugal particularly appreciated due to its freshness.

In this repo, I am predict the quality of Vinho Verde wines as to be of either low or high quality given the physicochemical components of wines that are introduced into the product during the wine-making process.  

My hypothesis is that the physicochemical properties of a wine (e.g. pH and alcohol content) which can be controlled at the winemaking stage, can be used to predict the overall quality of the wine. If this hypothesis is correct, then we can figure out which of the wine’s properties are most important in optimizing the output of high quality wines by a winemaker.

## Strategy
I use a blending strategy that combines several classifiers that are evaluated by their decision boundaries to create a decision boundary that best captures the class separation. 

<figure>
<p align="center">
  <img src="https://github.com/Pooret/Wine-Quality-Classification/blob/main/images/white_wine_results/SVC%20decision%20boundary%20(final%20est).png" alt="drawing" style="width:75%">
</p>
  
  <figcaption alighn = 'center'><b>Fig.1</b> - Decicion boundary for a stacking classifier model with SVC(kernel='rbf') as a meta estimator</figcaption>
  
  </figure>
