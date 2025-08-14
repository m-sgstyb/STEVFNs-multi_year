# STEVFNs-multi_year

This repository contains an adaptation of the Space-Time Energy Vector Flow Networks [(STEVFNs)](https://github.com/OmNomNomzzz/STEVFNs) model, developed to model multi-year energy systems.
As an application of the original software, this code is licensed under an MIT license (please see LICENSE file).

> [!IMPORTANT]
> If you use this adapted version of the STEVFNs framework in your research, please cite both:
> 
> 1. This repository and associated doctoral thesis (when published)
> 2. The original STEVFNs Tool software and the associated doctoral thesis by Aniq Ahsan
>
> The direct GitHub link to "Cite this repository" will only display the APA and BibTex citations for this repository. For the full citations list, please see or download the citations.bib file.

Please also see the NOTICE file to find the license for dependent software used in this model.

## Case Study
The code and data here present a case study for long-distance HVDC interconnections for renewable energy trade in America, with a specific focus on Mexico, USA (Western Interconnection), and Chile. 
## Documentation
Assumptions and discussion of the data can be found in the thesis.
## Installation
I recomment the use of conda package manager, installation instructions for different operating systems can be found  [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). If you wish to use this version of the software, you may clone this repository through your terminal as:

```
git clone https://github.com/m-sgstyb/STEVFNs-DPhil-MSB.git
```

Once Conda is installed and the repository cloned in your desired path locally, navigate to the cloned STEVFNs-multi_year folder in your terminal and run 
```
conda env -f create environment.yaml
```
The envrionment.yaml file containts the minimum dependencies included, as well as an installation of the Spyder IDE.

The default opnen-source optimiser CLARABEL will also be installed through the envrionment.yaml file. CVXPY allows for customisation and using different optimisation software. Other open-source options include SCS and ECOS and may also be used depending on your needs. Private optimisation software such as MOSEK can also be used with cvxpy with the appropriate license (see [CVXPY's solver features](https://www.cvxpy.org/tutorial/solvers/index.html))

