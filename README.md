## Welcome to Model Predictive Control for PyPSA

This repository implements model predictive control using the PyPSA's power flow as the optimization backend in each iteration of the rolling horizon.

This is a work in progress and only contains essential features.

### How it works

Control is based on the interplay of three main components: the __pypsa.Network__, containing the system configuration, multiple objects of the __Prophet__-type, responsible for returning time series to the pypsa components and the __Controller__, rolling snapshots forward in time while storing optimization outcomes.

### Installation

The repository has minimal requirements and its intended to stay this way. To install the dependencies run

```shell
conda env create -f environment.yaml
```

Additionally, a linear solver is needed. For this, please refer to the [PyPSA installation guide](https://pypsa.readthedocs.io/en/latest/installation.html).

### Examples

A minimal but illustrative example is presented in our [readthedocs](https://district-control.readthedocs.io/en/latest/). Feel free to take a look!

##### Contact

I am always happy about feedback and for people to contribute, feel free to reach out at <lukas.franken@ed.ac.uk>
