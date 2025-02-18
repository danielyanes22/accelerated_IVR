![repo version](https://img.shields.io/badge/Version-v.1.0-green)
![python version](https://img.shields.io/badge/python-3.12.1-blue)
![license](https://img.shields.io/badge/License-CC--BY-red)


<h2 id="Title">A machine learning pipeline to accelerate the design of *in vitro* release tests from liposomes</h2>

**Daniel Yanes**<sup>1</sup>, **Vasiliki Paraskevopoulou**<sup>2</sup>, **Heather Mead**<sup>2</sup>, **James Mann**<sup>2</sup>, **Magnus Röding**<sup>3,4</sup>, **Maryam Parhizkar**<sup>5</sup>, **Cameron Alexander**<sup>1</sup>, **Jamie Twycross**<sup>6*</sup>, **Mischa Zelzer**<sup>1*</sup>

<sup>1</sup>School of Pharmacy, University of Nottingham, University Park Campus, Nottingham, NG7 2RD, UK\
<sup>2</sup>Global Product Development, Pharmaceutical Technology & Development, Operations, AstraZeneca, Macclesfield, SK10 2NA, UK\
<sup>3</sup>Sustainable Innovation & Transformational Excellence, Pharmaceutical Technology & Development, Operations, AstraZeneca, Gothenburg, 43183 Mölndal, Sweden\
<sup>4</sup>Department of Mathematical Sciences, Chalmers University of Technology and University of Gothenburg, 41296 Göteborg, Sweden\
<sup>5</sup>School of Pharmacy, University College London, 29-39 Brunswick Square, London, WC1N 1AX, UK\
<sup>6</sup>School of Computer Science, University of Nottingham, Jubilee Campus, Wollaton Road, Nottingham, NG8 1BB, UK\
<sup>\*</sup>Corresponding authors: mischa.zelzer@nottingham.ac.uk; jamie.twycross@nottingham.ac.uk

**Abstract**\
Liposomes are amongst the most promising and versatile nanomedicine products employed in recent years. In vitro release (IVR) tests are critical during development of new liposome-based products. The drug release characteristics of a formulation are affected by multiple factors related to the formulation itself and the IVR method used. While the effect of some of these parameters has been explored, their relative importance and contribution to the final drug release profile are not understood sufficiently to enable efficient rational design choices. This causes delays in the development and approval of new medicines. In this study, a machine learning pipeline is developed, which can be used to better understand patterns in liposomal formulation properties, IVR methods and resulting drug release characteristics. A comprehensive database of liposome release profiles, including formulation properties, IVR method parameters and drug release is compiled from academic publications. A multiclass classification model is developed to predict the release profile type (kinetic class), with a significant increase in the balanced accuracy test score compared to a random baseline. The resulting machine learning approach enhances understanding of the complex liposome drug release process and provides a predictive tool to accelerate the design of liposomal IVR tests.   

**Graphical Abstract**\
![Figure 1](figures/ML_graphical_abstract.png?raw=true "Graphical Abstract")


<!-- Prerequisites-->
<h2 id="Prerequisites">Prerequisites</h2>
The following key Python packages are required to reproduce the analysis and results in the manuscript.
- [Pandas](https://pandas.pydata.org/) (2.1.4)
- [Numpy](https://numpy.org/) (1.26.4)
- [XGBoost](https://xgboost.readthedocs.io/) (2.0.3)
- [Scikit-learn](https://scikit-learn.org/) (1.4.0)
- [SciPy](https://docs.scipy.org/doc/) (1.15.1)
- [SHAP](https://shap.readthedocs.io/en/latest/) (0.44.1)


<h2 id="Installation">Installation</h2>
Install dependencies from the provided ``` requirements.txt ``` file. The code was tested on Microsoft Windows 10, Version 22H2.

```
pip install -r requirements.txt
```

<!-- Content-->
<h2 id="content">Content</h2>
This repository is structured in the following way:


<!-- How to cite-->
<h2 id="How-to-cite">How to cite</h2>


<!-- License-->
<h2 id="License">License</h2>

This codebase is under a CC-BY license. For use of specific models, please refer to the model licenses found in the original 
packages.