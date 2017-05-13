# adml_dataminer
adml_dataminer

## Dependencies
### python
- numpy==1.12.1
- ml_metrics==0.1.4
- pandas==0.19.2
- scipy==0.19.0
- matplotlib==2.0.0
- local==0.0.0
- scikit_learn==0.18.1
- statsmodels==0.8.0
- xgboost==0.6

## example can be viewed [example]()

### 分享
- 任何想法分享到google drive中
- https://drive.google.com/drive/folders/0B8zqrhAmm5-1VGpSWkN6VW9lc00
### Ensemble 
1.sklearn.ensemble.VotingClassifier
ensemble predicting result from different base model(svm, linear, rf, xgb)
2.stacking 
Assigning different weights for base model predicting results, the second layer model is created
to link first layer results and original target.(robost/stronger)
 

