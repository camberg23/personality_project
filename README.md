Welcome! To try out any of our AutoML TPOT pipelines, download both the .py and the .csv file that share a name (though the py file may also be followed by an irrelevant number, e.g., "extentpc3"). Edit the code (as needed) such that the column you are 
attempting to assess corresponds with the appropriate column in the csv. If a file name is followed by "classify," it means that the 100-point scale across which the participant identified themselves was bifurcated into the group above the average value and the group below the average value for that particular scale. If a file name is followed by "pole," it means that individuals approaching the either end of the distribution on the 100-point scale were compared against each other. (As of November, 2020, many of these files haven't yet been uploaded.) Recommendation: vary the random seed frequently to get a feel for the range of
outcomes, keeping in mind the the CV score commented into the code signifies the average value that the particular pipeline was able to achieve on the dataset.


Sources, references, and tools used in the creation of these piplines:

Randal S. Olson, Nathan Bartley, Ryan J. Urbanowicz, and Jason H. Moore (2016). Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science. Proceedings of GECCO 2016, pages 485-492.

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
