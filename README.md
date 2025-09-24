![repo version](https://img.shields.io/badge/Version-v.1.0-green)
![python version](https://img.shields.io/badge/python-3.12.1-blue)
![license](https://img.shields.io/badge/License-CC--BY-red)


<h2 id="Title">A machine learning workflow to accelerate the design of <em>in vitro</em> release tests from liposomes</h2>


**Daniel Yanes**<sup>1</sup>, **Vasiliki Paraskevopoulou**<sup>2</sup>, **Heather Mead**<sup>2</sup>, **James Mann**<sup>2</sup>, **Magnus Röding**<sup>3,4</sup>, **Maryam Parhizkar**<sup>5</sup>, **Cameron Alexander**<sup>1</sup>, **Jamie Twycross**<sup>6*</sup>, **Mischa Zelzer**<sup>1*</sup>

<sup>1</sup>School of Pharmacy, University of Nottingham, University Park Campus, Nottingham, NG7 2RD, UK\
<sup>2</sup>Global Product Development, Pharmaceutical Technology & Development, Operations, AstraZeneca, Macclesfield, SK10 2NA, UK\
<sup>3</sup>Sustainable Innovation & Transformational Excellence, Pharmaceutical Technology & Development, Operations, AstraZeneca, Gothenburg, 43183 Mölndal, Sweden\
<sup>4</sup>Department of Mathematical Sciences, Chalmers University of Technology and University of Gothenburg, 41296 Göteborg, Sweden\
<sup>5</sup>School of Pharmacy, University College London, 29-39 Brunswick Square, London, WC1N 1AX, UK\
<sup>6</sup>School of Computer Science, University of Nottingham, Jubilee Campus, Wollaton Road, Nottingham, NG8 1BB, UK\
<sup>\*</sup>Corresponding authors: mischa.zelzer@nottingham.ac.uk; jamie.twycross@nottingham.ac.uk

**Abstract**\
Liposomes are amongst the most promising and versatile nanomedicine products employed in recent years. In vitro release (IVR) tests are critical during development of new liposome-based products. The drug release characteristics of a formulation are affected by multiple factors related to the formulation itself and the IVR method used. While the effect of some of these parameters has been explored, their relative importance and contribution to the final drug release profile are not understood sufficiently to enable efficient rational design choices. This prolongs the development and approval of new medicines. In this study, a machine learning workflow is developed, which can be used to better understand patterns in liposome formulation properties, IVR methods and the resulting drug release characteristics. A comprehensive database of liposome release profiles, including formulation properties, IVR method parameters, and drug release profiles is compiled from academic publications. A classification model is developed to predict the release profile type (kinetic class), with a significant increase in the balanced accuracy test score compared to a random baseline. The resulting machine learning approach enhances understanding of the complex liposome drug release dynamics and provides a predictive tool to accelerate the design of liposome IVR tests.    

**Graphical Abstract**\
![Figure 1](figures/ML_graphical_abstract.png?raw=true "Graphical Abstract")


<!-- Prerequisites-->
<h2 id="Prerequisites">Prerequisites</h2>

The following key Python packages are required to reproduce the analysis, results, and figures in the manuscript:

- [Pandas](https://pandas.pydata.org/) (2.1.4)  
- [Numpy](https://numpy.org/) (1.26.4)  
- [XGBoost](https://xgboost.readthedocs.io/) (2.0.3)  
- [Scikit-learn](https://scikit-learn.org/) (1.4.0)  
- [SciPy](https://docs.scipy.org/doc/) (1.15.1)  
- [SHAP](https://shap.readthedocs.io/en/latest/) (0.44.1)  


<h2 id="Installation">Installation</h2>
Install dependencies from the `requirements.txt` file. The code was tested on Microsoft Windows 10, Version 22H2.

```
pip install -r requirements.txt
```

<!-- Content-->
<h2 id="content">Project structure</h2>
This following folder structure gives an overview of the repository:

<pre>
├── data/
│   ├── clean/  # datasets for ML classifier screening on 7 and 9 features
│   ├── unprocessed/ # fitted weibull params with f2 scores > 50 & backend datasets 
├── experiments/ # Run the scripts in this order specified in __init__.py file to reproduce the analysis
├── models # .pkl file of each ML models trained on 7 features using stratified 5-fold cross validation
├── results # clustering, kinetic model fitting and ML_classifier evaluation files 
├── src # helper functions for running batch parameter estimation and data preprocessing 
</pre>

<h2 id="content">Running the models and analysis</h2> 
Run the scripts in the order specified in 'experiments/__init__.py'.

<!-- How to cite-->
<h2 id="How-to-cite">Paper link</h2>
https://doi.org/10.1039/D5DD00112A

<!-- License-->
<h2 id="License">License</h2>
This codebase is under a CC-BY license. For use of specific models, please refer to the model licenses found in the original 
packages.