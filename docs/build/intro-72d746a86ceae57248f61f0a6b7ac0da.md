# 👋 Welcome !

This [Jupyter book](https://jupyterbook.org) was created by [Clement Hardy](https://klemet.github.io/) for the [research theme 4](https://diverseproject.uqo.ca/theme-4-evaluation-various-forest-management-approaches/) of the [DIVERSE project](https://diverseproject.uqo.ca/).

## 🎯 Goal of this documentation

The objective here is to calibrate the parameters of [PnET Succession extension](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession) for the [LANDIS-II model](https://www.landis-ii.org/), in the context of the different landscape that will be simulated for the [DIVERSE project](https://diverseproject.uqo.ca/).

PnET-Succession is one of the most complex extensions of LANDIS-II, based on the PnET familly of eco-physiological models. To be certain that the vegetation dynamics that it simulates will be realistic, we need to calibrate it carefully.

The [PnET succession User Guide](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession/blob/master/deploy/docs/LANDIS-II%20PnET-Succession%20v5.1%20User%20Guide.pdf) contains a lengthy section of calibration tips by [Eric Gustafson](https://research.fs.usda.gov/about/people/gustafson) that I will use as a guide for this calibration process.

```{admonition} Making the calibration of PnET easier for everyone
:class: tip
One of the main goals of this documentation is to translate the calibration tips of Eric Gustafson into a replicable and entirely documented procedure. As such, any research team should find here a calibration of PnET Succession from A to Z that can be replicated and reused freely.
```

## 🪜 Steps of the calibration

There are many ways to calibrate a model like PnET-Succession. It can be difficult to choose which one to use. Here, I propose an approach based my own understanding of the [calibration tips of Eric Gustafson](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession/blob/master/deploy/docs/LANDIS-II%20PnET-Succession%20v5.1%20User%20Guide.pdf).

This approach goes through the following steps :

- Generating some initial values for the parameters (or taking them from the litterature) (notebook 3)
- Estimating growth curves in ideal conditions (monoculture, etc.) from empirical data and gathering different informations on our tree species (shade tolerance, etc.) (notebook 4)
- Gathering climate data representative of the study landscape and averaging it to create a "mild" version of the local climate (notebook 5)
- Calibrate the growth curve of each tree species in "ideal conditions" (soil with good water retention, mild climate and monoculture) using the growth curves, information and climate data generated at the previous steps (Notebook 6)
- Assessing the competition between the tree species and making adjustments if necessary (Notebook 7)
- Calibrating `MaxPest` to adjust the frequency of establishment of new age cohorts in the landscape (Notebook 8) 


## 📖 Format of this documentation

This documentation is a Jupyter Book (the HTML pages that you can read at [https://klemet.github.io/Calibration_PnET_DIVERSE/](https://klemet.github.io/Calibration_PnET_DIVERSE/)), composed of several files (markdown files and Jupyter notebooks) that you can find in [the Github repository that contains the book](https://github.com/Klemet/Calibration_PnET_DIVERSE).

The Jupyter notebooks can be run in a Docker container by using the files present in the Github repository, so as to reproduce the exact same environment with which the notebooks were executed. This ensures replicability on the long term.

The Jupyter notebooks can then be transformed into a HTML version using the script present in the notebook `BuildAndUpdateJupyterBook.ipynb`.

➡️ **📖 If you simply want to inform yourself as to the calibration process, just keep reading !**

➡️ **💻 If you want to run the code used for this calibration process on your own computer (for example, to calibrate new species for your own study area), read the instructions of [the readme of the repository](https://github.com/Klemet/Calibration_PnET_DIVERSE/blob/main/README.md)**. They will help you deploy a Docker image on your computer that will contain the entire environment (including LANDIS-II) used for this calibration within minutes. You can then edit the notebooks so as to calibrate the parameters in your own study area and with different species.

**📖 Happy reading !**

## ✒️ Table of contents

```{tableofcontents}
```
