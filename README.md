# Wine-Quality-Classification

## Introduction
Wine quality is a complicated metric that takes into account the wine’s color, terroir (i.e. it’s growing conditions), the balance, complexity, length of time on the palette, and the distinctiveness of the flavor notes of the wine (.e.g terms like oak, cherry, vanilla, and tobacco are common red wine notes, whereas whites tend to have citrus notes).

In this project, I have downloaded a dataset containing red and white vinho verde wines from Portugal for purposes of predicting wine quality based on physicochemical tests using machine learning.

To give a reason as to why this is important, I want to start with a quote from Maynard Anderw Amerine, who was a pioneering researcher in the cultivation, fermentation, and sensory evaluation of wine that helped to establish the metric upon which wines are judged.

                                      “Wines quality is easier to detect than define”
                                      
This makes sense, if you enjoy the flavor of a soda for example, it is very easy to say whether or not we enjoy it, but knowing why we enjoy it is another matter entirely.

Similarly, quality is primarily a subjective measurement strongly influenced by extrinsic factors. While there is somewhat of a general consensus among wine connoisseurs as to what constitutes wine quality, that is, it is a subjective measurement that comes from extensive experience in wine tasting.

There are some quantifiable aspects to this howerer, and for example in white wines, acidity and alcohol/sugar should match; if the acidity is not enough compared to the wine’s sugar content level, you run the risk of having a wine that is too cloying.
For reds, tannins, acidity and alcohol should all be in balance. Negative quality factors, such as off-odors, are generally easier to identify and control. Positive quality factors tend to be more elusive.

The hypothesis is that the physicochemical properties of a wine (e.g. pH and alcohol content) which can be controlled at the winemaking stage, can be used to predict the overall quality of a wine. If this hypothesis is correct, then we can figure out which of the wine’s properties are most important in optimizing the output of high quality wines by a winemaker.

### The Features

Vinho verde is a medium-alcohol wine from the Minho (northwest) region of Portugal particularly appreciated due to its freshness.
In this project, I am tasked with predicting the quality of Vinho Verde wines (ranging from 0 for the lowest quality wines to 10 for the highest quality wines), given the physicochemical components of wines that are introduced into the product during the wine-making process.

![Screen Shot 2022-11-16 at 4 40 40 PM](https://user-images.githubusercontent.com/64797107/202300407-7be2ed01-68d6-45e7-933a-40f7d9270bbb.png)

Tartaric acid is the primary acid in wine grapes. It’s probably the most durable acid in a wine, and it resists much
of the effects of other acids. That’s why it’s called a fixed acid. That makes it one of the most important parts in stabilizing a wine’s ultimate color and flavor profile.

Citric acid has a minor presence in wine, but a noticeable one nonetheless. The quantity of citric acid in wine is about 1/20th that of tartaric acid. It’s mostly added to wines after fermentation due to yeast’s tendency to convert citric acid to acetic acid. It has an aggressive acidic taste, is often added by winemakers to increase a wine’s total acidity, and should be added very cautiously.

Sulfur dioxide is added to wine during winemaking to prevent the microbial spoilage, oxidation and color changes due to undesirable enzymatic and non-enzymatic reactions. High concentrations of sulfur dioxide affect the final quality of the wine, mainly the smell and the taste and can inhibit the malolactic fermentation.

Sulfur dioxide (SO2) as potassium thiosulphate was used as a microbial growth inhibitor during the current fruit juice substrate preparation to inhibit the growth of some yeast species and the majority of bacteria related to wine spoilage. Due to antiseptic and antioxidant properties on the final wine, sulfur dioxide (SO2) is the most versatile and efficient additive than other additives such as dimethyl dicarbonate (DMDC) used during winemaking.

### Business Case

By being able to predict a wine’s quality from a given set of physicochemical parameters, wine-makers can reasonably produce high quality wines with the given features as inputs to the model. This can help a wine-maker to plan ahead to produce a high quality wine as well, by using the predicted parameters to obtain the best physicochemical values as predicted by the model.

## Data Wrangling

The wine data data set was available in two .csv files from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/ml/datasets/wine+quality), split into red and white wines. The datasets were loaded into pandas DataFrames and examined.

### Dealing with Data Size Issues
To best reduce bias, I want to set aside a portion of the test data as early as possible. However, I can see there are some issues that need to be addressed first due to the size of the data.

![Screen Shot 2022-11-16 at 4 43 27 PM](https://user-images.githubusercontent.com/64797107/202300855-0a6bdf84-25a5-4f43-9bd1-104329219b0b.png)

The target for this classification problem is the quality feature of the dataframe with scores ranging from 0 - 10 (bad - good). Plotting the distribution of these data as shown in Figure 1 shows that I am limited by the red wine data size, and that the target data values need to be categorized. As I need to classify the wines into high and low qualities, I decide that any wine with a quality score of 7 or greater will define the decision boundary between high and low quality wines. Figure 2 shows the distributions of counts for each of these two classes, and that there is a class imbalance that also needs to
be addressed.

![Screen Shot 2022-11-16 at 4 43 59 PM](https://user-images.githubusercontent.com/64797107/202300963-07546efd-a5b5-4f76-a809-32b2be818cb4.png)

I decided to set aside 20% of the data for the white wines, and 40% of the data for the red, stratifying the target class. My reasoning behind this is that the test data will have 87 samples of the minority class (high quality wines) at this high of a split, and due to the limited data, I need to have enough to train a model that can recognize the minority class. The test data are then saved and the current data are used for data exploration, model training, and validation.

## Dataset Features

Each dataset has 11 features and 1 target, where the features are physicochemical descriptors of the wines. These data were published by [Cortez et al](https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub), of red and white vinho verde samples from Portugal for classification purposes.

### Data Discrepancies

All of the features are of type float and don’t contain any missing data. However, roughly 20% of the data are duplicated. In general, [it is good practice to remove duplicated data on the training set so that the model can better generalize to the full dataset.](https://www.quora.com/Should-we-remove-duplicates-from-a-data-set-while-training-a-Machine-Learning-algorithm-shallow-and-or-deep-methods) I can always test whether or not duplicates affect model performance down the pipeline, but for now I will remove them moving forward.

I next wanted to look at outliers in the data. Figure 3 shows a boxplot of the red and white wine densities with clear outliers present in the red wines dataset. For visualization purposes I will remove them to better understand the spread of the data3. However I plan to wait on removing them for modeling purposes until after I do feature selection.

![Screen Shot 2022-11-16 at 4 46 56 PM](https://user-images.githubusercontent.com/64797107/202301459-43229ebc-262a-46c7-9056-5adbf1221eda.png)

### Target-Feature Relationships

I wanted to see if the target feature had any strong dependencies from the physicochemical features in the dataset. This could allow me to better understand the differences between high and low quality wines as well as to aid in model training and prediction.

Figure 4 shows a violin plot of red and white wine quality vs alcohol content. It is readily apparent that high quality wines have a higher alcohol content than lower quality wines. The differences in the means of the data for each class (as indicated by the dashed - - lines). Shows how readily apparent this difference is, with the average high quality of red wines being a bit higher than that of the white wines.

![Screen Shot 2022-11-16 at 4 47 38 PM](https://user-images.githubusercontent.com/64797107/202301593-4e9be5fd-7222-47ea-9b71-46b8b05b3dc2.png)

### Feature-Feature Relationships
Some of the features are obviously linear combinations of other features in the dataset. The density feature for example, is given by Equation 1.

                                                    Eq. 1 𝜌 = 𝑚 ÷ 𝑉
                                        𝜌 = Density of the solution in 𝑔/𝑐𝑚3
                                       𝑚 = mass of the solution in grams (𝑔)
                                 𝑉 = Volume of the solution in cubic centimeters (𝑐𝑚3)
                                 
As 9 of the 11 features in the dataset are are the chemicals given in weight-per-volume dissolved in the wine, these features will naturally be collinear with density. Figure 5 is a heatmap of the pearson correlations between the features with a min heatmap visualization threshold of pearson = 0.2.

![Screen Shot 2022-11-16 at 4 49 45 PM](https://user-images.githubusercontent.com/64797107/202301970-86b5661d-c1fc-4e5d-b68c-64133aa7b0e2.png)

For both red and white wines, it's easy to see that density is highly correlated with residual sugar in the white wines, and fixed acidity in the red wines. As the density of the wine represents the total weight contribution of all the components dissolved in the wine, it can be
inferred that the sugar makes up a greater overall weight percentage of white wines than in red wines, and that the same can be true for the fixed acidity for red wines over that of white. I can’t say whether or not these features are more or less important as they are simply weight measurements, but it’s possible that they may be of greater significance given their greater abundance in each of the wines. I will look at feature importances next to get a general idea of the wines.

### Feature Importance

Given the problem, it would be advantageous for the employer to know which of the physicochemical features in the dataset contribute the most to model predictions. I will aim to use permutation feature importance using the best models selected for each dataset. Before that is done, I’d like to get a quick idea as to how a Random Forest Classifier will predict feature importances on the training dataset.

#### Red Wines
Figure 6 shows the importances for the red wines dataset after alcohol and density were removed. Alcohol was removed as it is by far the most important feature and thus dwarfs other features. Density was removed as multicollinear with many other features in the dataset.

![Screen Shot 2022-11-16 at 4 51 27 PM](https://user-images.githubusercontent.com/64797107/202302254-1148e9e3-05dc-45ff-93e5-b45ad07fbb2e.png)

The most important features predicted on the training data by a random forest classifier for the red data are sulfates, volatile acidity, and total sulfur dioxide.

#### White Wines

The features that were removed from the red wines dataset were also removed from the white wines, and feature importance was
similarly predicted on the training data using a random forest classifier, shown in Figure 7.

![Screen Shot 2022-11-16 at 4 52 21 PM](https://user-images.githubusercontent.com/64797107/202302394-6248be95-6ed4-4818-bbaa-7e0a5f8fd93b.png)

The model predicts chlorides to be most important, which comes from salt in the wine. I expected residual sugar and volatile acidity to be here, but I will see how the final model’s predictive features' importances compare.

## Modeling
My strategy in addressing the problem of classifying each of the wine datasets by their quality classes was to develop a machine learning pipeline that would allow me to quickly prototype several classification models, as well as allowing me to experiment with various data preprocessing strategies and hyperparameter tuning.

At the end of this pipeline, I will blend each of the models that I’ve trained to obtain the best performance across the range of models tested. This throw everything and see what sticks approach is a nice way to optimize performance as well as to gain insights into the modeling by seeing how each of the models do, comparatively.

Frank Ceballos provides a [well-written and comprehensive article on stacking classifiers and their uses in boosting performance](https://towardsdatascience.com/stacking-classifiers-for-higher-predictive-performance-566f963e4840). They are best used when an increase in performance can’t be achieved through any further data acquisition, feature engineering, or preprocessing and all typical classification model approaches have been exhausted. The downside to these types of models is that they can be computationally prohibitive. Luckily, I am dealing with fairly small datasets where this approach makes sense.

### Choice of Metrics

performance. I will select the final model based off of the precision-recall scores and whatever is appropriate for the situation (e.g. excluding models with very long fit times, etc.).

### Model Pipeline

![Screen Shot 2022-11-16 at 4 54 07 PM](https://user-images.githubusercontent.com/64797107/202302683-ba5f6e09-3ecc-4ff8-883a-7ce18e255b0c.png)

The general pipeline for the wine classifications shown above in Figure 8. This approach uses a stacking classifier that blends the predictions made from several other classifiers by having those predictions passed as inputs for a final estimator (a gradient boosting classifier in this case).

The data were preprocessed by first undergoing feature selection and then applying Yeo-Johnson transformations for the linear estimators, otherwise the standard scaler was used to normalize the variance in the feature data.

## Results and Discussion

The test roc auc test scores for each of the classifiers was evaluated by a 10-fold cross validation. The results are shown below in Figure 9.

![Screen Shot 2022-11-16 at 4 54 56 PM](https://user-images.githubusercontent.com/64797107/202302810-5540e346-f6de-401e-9de8-dcc18e44bba9.png)

The classifiers with the best mean test roc auc scores are the stacking classifiers in both datasets, which is to be expected. However, the best evaluator for the red wines data uses a support vector machine5 classifier for its final estimator, whereas the white wines feature a gradient boosting classifier.

High and low quality wines in the white dataset are not so readily linearly separable and are therefore more difficult for a linear classifier to classify. Because the KNN classifier uses localized proximity to make predictions, localized groupings of high quality wines are able to be better predicted.

Plotting the decision boundaries and evaluating the models provides insight into how the model performs as well as how to tune it in a more intuitive way.

### Red Wines

The mean results for the classification models are presented in Table 2 with roc auc train and test scores and modeling times. What is immediately apparent is that the higher performing models are those that tend to form a single, continuous decision boundary-one in which there is a clear distinction between the predicted positive and negative classes. Estimators that form many small, discrete decision boundaries tend to perform more poorly in relation.

![Screen Shot 2022-11-16 at 4 55 31 PM](https://user-images.githubusercontent.com/64797107/202302921-aa54895a-a916-4849-9ab7-d0495498202f.png)

#### Stacking Classifier Comparisons

The stacking classifier takes the predictions of the estimators and sends them as inputs for a final estimator. This results in the final estimator’s decision boundary that “blends” features present in the input estimators’ decision boundaries. Appendix A1 # shows the decision boundaries for red wines following PCA decomposition into two principal components for visualization. The SVM final estimator predicts a separation between high quality wines and low quality wines in the 2D-plane, and the randomness associated in the boundary are from the input estimators.

Conversely, the gradient boosting classifier creates multiple islands that lend themselves to a lower ROC AUC score, but overall performs better in precision by predicting a higher ratio of high quality wines over low ones than the svc final estimator. It should be noted that stacking classifiers tend to overfit as can be seen in Table 1, and reducing the complexity of the model by dropping out some of the input estimators, or introducing some form of regularization into the final estimator will be the next step.

#### Feature Importance

I elected to use the stacking classifier with the gradient boost as the final estimator in examining the feature importance. Note that the negative values in Figure 10 are features the model deems unimportant in through permutation feature importance [that can appear in smaller datasets](https://www.kaggle.com/dansbecker/permutation-importance). The most important features according to the model are alcohol, followed by sulfates and volatile acidity.

![Screen Shot 2022-11-16 at 4 57 22 PM](https://user-images.githubusercontent.com/64797107/202303287-b09cb4d6-43d5-418b-bac9-47761f7c6c7c.png)

### White Wines

Table 3 has the mean ROC AUC scores for the test data in the white wines. Compared to the red wines dataset, the white wines is 3x larger, and possibly as a result there is greater overlap between the high and low quality wines see Appendix B. Contrasting this as well, classifiers on the white wines dataset that tend to perform well are those that form multiple less-defined decision
boundaries instead of the well-defined ones that did well in the red wines dataset.

![Screen Shot 2022-11-16 at 4 58 55 PM](https://user-images.githubusercontent.com/64797107/202303484-611a257d-f93e-451f-ab87-4a206be9eda6.png)

### Stacking Classifier Comparisons

To get a better idea as to why this idea works, I turn
again to the Appendix B where the decision boundaries for the two stacking classifiers for the white wines are plotted following PCA decomposition for ease of visualization (Figures B1, B3). Using SVM as a final estimator creates a decision boundary that subdivides the plane into high quality and low quality wines. While this plot looks reasonable, there are still many misclassifications for both classes that results in a decent recall (as some high quality wines are found in the low quality region), and moderate precision (there are far too many low quality wines in the high quality class region).

Using the gradient boosting classifier as a final estimator boosts the precision to 68%. The downside of this is that the fit time for the model is 10x longer than the already long svc stacking classifier. This highlights the limitation of the stacking classifier in that it is computationally more expensive than the other classifiers used. What is seen is what was advertised, however, and using it results in a 3% in the roc auc test score after cross-validation.

#### Feature Importance

Permutation feature importance was done using the stacking classifier with the gradient boost final estimator. Figure 11 shows that alcohol is the most important feature followed by pH, sulphates, free sulfur dioxide, and residual sugar. I was surprised to see that the chlorides were not considered as important to the model as the random forest indicated in Figure 7. Many of the features have similar scores and may indicate that the model needs to be further tuned or I need to revise my feature selection strategy.

![Screen Shot 2022-11-16 at 5 00 28 PM](https://user-images.githubusercontent.com/64797107/202303779-58d0509d-3888-497f-b133-edb86140a0cb.png)

## Conclusions

The models I built for the client predict the wine qualities of red and white wines from Portugal given physicochemical properties of the wines, and these models are used to assess the importance of the physicochemical features using permutation feature importance.

For red wines, the most important qualities are the alcohol content and the sulfates in the wine. [Sulfates are added to wines to prevent spoilage](https://www.thegrapeculture.com/blog/does-wine-contain-sulfates-and-does-it-matter), and as the third most important feature, volatile acidity, is an indicator of spoilage, the most important features for high quality red wines are those that affect wine spoilage.

For white wines, the model predicts that alcohol is the most important, followed by pH and residual sugar. I am less confident about these predictions than I am of the red due to the complexity of the model and the more complicated dataset for the white wines. Despite this, the stacking classifier with a gradient boosting final estimator has a relatively high precision score of 0.69 on the test dataset, meaning wines predicted to be of high quality will be correct on average ~70% of the time. In comparison, the best red wine stacking classifier would be making those same predictions ~50% of the time.

The stacking models were used because they can boost the performance of a model. And while that is seen with the white wines dataset, they have 10-100x longer fit times and tend to overfit the data. If I were to present this to my client, I would need to make sure that these are issues that need to be addressed. It would be feasible if a 3% increase in the precision score of the model used would forecast profits that justify the operating costs.

Next Steps  
  ● Apply a neural network and see how it compares  
  ● Review feature importances for white wines model  
  ● Improve model precision scores for red wines  
  ● Try and get more data from client  
  ● Do a cost-benefit-analysis on model implementation  
  
## Appendices

### Appendix A: Red Wines

![Screen Shot 2022-11-16 at 5 02 34 PM](https://user-images.githubusercontent.com/64797107/202304170-3e71a18f-9039-4aa8-b749-32593c014a88.png)


### Appendix B: White Wines

![Screen Shot 2022-11-16 at 5 03 34 PM](https://user-images.githubusercontent.com/64797107/202304284-c896d6ec-e533-4d48-b689-0299c598f722.png)

