# Wine-Quality-Classification

**Project Description**  
Vinho verde is a medium-alcohol wine from the Minho (northwest) region of Portugal particularly appreciated due to its freshness.

In this repo, I am predict the quality of Vinho Verde wines as to be of either low or high quality given the physicochemical components of wines that are introduced into the product during the wine-making process.  

My hypothesis is that the physicochemical properties of a wine (e.g. pH and alcohol content) which can be controlled at the winemaking stage, can be used to predict the overall quality of the wine. If this hypothesis is correct, then we can figure out which of the wine’s properties are most important in optimizing the output of high quality wines by a winemaker.

## Strategy
I use a blending strategy that combines several classifiers that are evaluated by their decision boundaries to create a decision boundary that best captures the class separation. 


<p align="center">
  <figure>
    <img src="https://github.com/Pooret/Wine-Quality-Classification/blob/main/images/red_wine_results/stacking_dark_svm.png" alt="drawing" width=500>
    <figcaption alighn = 'center'><b>Fig.1</b> - Decicion boundary for a stacking classifier model with SVC(kernel='rbf') as a meta estimator</figcaption>
  </figure>
</p>
  
  
## Results
The model for the red wines as shown above outperforms the other classifiers in its ROC AUC performance on the test data. The stacking classifier is computationally more expensive, however, which is the trade-off that comes with it's potential peformance boost.

<p align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/Pooret/Wine-Quality-Classification/main/images/red_wine_results/red%20wine%20classifiers%20all.png" alt="drawing" width=400>
    <figcaption alighn = 'center'><b>Fig.2</b> - ROC_AUC test scores for red wines classifiers</figcaption>
  </figure>
</p>
  
 
