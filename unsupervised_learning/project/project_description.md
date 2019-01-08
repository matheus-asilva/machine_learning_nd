# Content: Unsupervised Learning
## Project: Creating Customer Segments
### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Code

The main code for this project is located in the `finding_donors.ipynb` notebook file.

### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython notebook finding_donors.ipynb
```  
or
```bash
jupyter notebook finding_donors.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data
The customer segments data is included as a selection of 440 data points collected on data found from clients of a wholesale distributor in Lisbon, Portugal. More information can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers).

Note (m.u.) is shorthand for _monetary units_.

Features

1. `Fresh`: annual spending (m.u.) on fresh products (Continuous);
2. `Milk`: annual spending (m.u.) on milk products (Continuous);
3. `Grocery`: annual spending (m.u.) on grocery products (Continuous);
4. `Frozen`: annual spending (m.u.) on frozen products (Continuous);
5. `Detergents_Paper`: annual spending (m.u.) on detergents and paper products (Continuous);
6. `Delicatessen`: annual spending (m.u.) on and delicatessen products (Continuous);
7. `Channel`: {Hotel/Restaurant/Cafe - 1, Retail - 2} (Nominal)
8. `Region`: {Lisbon - 1, Oporto - 2, or Other - 3} (Nominal)
