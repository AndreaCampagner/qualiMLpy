# qualiMLpy
This a repository collecting all metrics, algorithms and pieces of code related to data quality
for Machine Learning, developed by me and others at the MUDI lab (https://www.mudilab.net/mudi/) of
the DISCo dept. @ University of Milano-Bicocca.

Code dependencies (also depends on which files you want to use) are: scikit-learn, numpy, scipy, matplotlib.

Some of our code has also been provisionally deployed in the form of web sandbox tools:

- Degree of correspondence: https://reprdeg-test.herokuapp.com/
- Degree of concordance: https://reliability-test.herokuapp.com/

You can find more information on most of these metrics and code in various articles:

* Degree of correspondence (correspondence.py) and degree of correspondence (reliability.py)

    * Cabitza, F., Campagner, A., & Sconfienza, L. M. (2020). As if sand were stone. New concepts and metrics to probe the ground on which to build trustable AI. BMC Medical Informatics and Decision Making, 20(1), 1-21.

    * Cabitza, F., Campagner, A., Albano, D., et al. (2020). The elephant in the machine: Proposing a new metric of data reliability and its application to a medical case to assess classification reliability. Applied Sciences, 10(11), 4014.

* H-accuracy (ha.py)

    * Campagner, A., Sconfienza, L., & Cabitza, F. (2020). H-accuracy, an alternative metric to assess classification models in medicine. Digital Personalized Health and Medicine; Studies in Health Technology and Informatics; IOS Press: Amsterdam, The Netherlands, 270.

* Meta-validation Methodology plots (step_one.py + step_two.py)

    * Cabitza, F., Campagner, A., Soares, F., de Guardiana Romualdo, L. G., Challa, F., Sulejmani, A., Seghezzi, M., Carobene, A. (2021). The Importance of Being External. Lessons learnt from the external validation of a machine learning model for COVID-19 diagnosis across 3 continents. Computer Methods and Programs in Biomedicine (Submitted)

In this repository you can also find some general utilities that are not directly authored by me (or anyone @ MUDI lab), for which we
provide a easy-to-use Python implementation (mostly compatible with the Python data science stack). For more info, please refer to the
following publications:

* Riley, R. D., Debray, T. P., Collins, G. S., Archer, L., Ensor, J., van Smeden, M., & Snell, K. I. (2021). Minimum sample size for external validation of a clinical prediction model with a binary outcome. Statistics in Medicine.
* Bradley, A. A., Schwartz, S. S., & Hashino, T. (2008). Sampling uncertainty and confidence intervals for the Brier score and Brier skill score. Weather and Forecasting, 23(5), 992-1006.

