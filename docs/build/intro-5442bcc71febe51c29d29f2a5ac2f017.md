# 👋 Welcome !

This [Jupyter book](https://jupyterbook.org) was created by [Clement Hardy](https://klemet.github.io/) for the [research theme 4](https://diverseproject.uqo.ca/theme-4-evaluation-various-forest-management-approaches/) of the [DIVERSE project](https://diverseproject.uqo.ca/).

## 🎯 Goal of this documentation

The objective here is to calibrate the [PnET Succession extension](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession) for the [LANDIS-II model](https://www.landis-ii.org/), in the context of the different landscape that will be simulated for the [DIVERSE project](https://diverseproject.uqo.ca/).

PnET-Succession is one of the most complex extensions of LANDIS-II, based on the PnET familly of eco-physiological models. To be certain that the vegetation dynamics that it simulates will be realistic, we need to calibrate it carefully.

The [PnET succession User Guide](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession/blob/master/deploy/docs/LANDIS-II%20PnET-Succession%20v5.1%20User%20Guide.pdf) contains a lengthy section of calibration tips by [Eric Gustafson](https://research.fs.usda.gov/about/people/gustafson) that I will use as a guide for this calibration process.

```{admonition} Making the calibration of PnET easier for everyone
:class: tip
One of the main goals of this documentation is to translate the calibration tips of Eric Gustafson into a replicable and entirely documented procedure. As such, any research team should find here a calibration of PnET Succession from A to Z that can be replicated and reused freely.
```

## 🪜 Steps of the calibration

There are many ways to calibrate a model like PnET-Succession. It can be difficult to choose which one to use. Here, I propose an approach based my own understanding of the [calibration tips of Eric Gustafson](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession/blob/master/deploy/docs/LANDIS-II%20PnET-Succession%20v5.1%20User%20Guide.pdf).

This approach is a multi-step procedure that focuses on calibrating certain parameters one after the others, in simulated conditions of increasing complexity. By doing this, we can isolate as much as possible the counfounding effect of different parameters and focus on one type of output/outcome at a time while keeping other parameters fixed.

The different phases are :

- Phase 1 🌳 : Calibration of species in optimal conditions (monoculture + mesic soil + average and unchanging climate).
    - Sub-phase 1 : Calibration of the peak of the growth curve
    - Sub-phase 2 : Calibration of the initiation of the growth curve
    - Sub-phase 3 : Calibration of the senescence (end of the growth curve)
- Phase 2 🌤️ : Calibration of the competition for light (mixed stands + mesic soil + average and unchanging climate) via a matrix of expected light competition outcomes between pairs of species. A calibration might also be done based on the expected outcomes of known combination of species more complex than a pair (e.g. species assemblages typical of the region) to see if PnET can replicate the cohabitation of these species together.
- Phase 3 🌧️ : Calibration of the absorption of water (monoculture stands + average and unchanging temperature + conditions of waterlogging or drought) via a matrix of expected outcomes for each species in conditions of drought or waterlogging (i.e. regular growth, decrease in growth, strong decrease in growth, death).
- Phase 4 🌡️ : Calibration of the effect of the temperature (monoculture stands + changing temperatures + mesic soil and average precipitations) via a matrix of expected outcomes for each species in conditions of increasing temperatures (i.e. regular growth, decrease in growth, strong decrease in growth, death). Landscape-scale simulation might also be used to see if expected processes (e.g. increase in deciduous species, decrease in coniferous species) is observed at this larger scale.
- Phase 5 🌱 : Calibration of landscape-scale establishment (one species at a time in the landscape to simulate + average temperatures and precipitations) by monitoring the number of cohorts produced at each time step via dispersion/reproduction of the species, and comparing to expected values.

**Phase 1 will most likely be the most time-consuming and important**, as this is where we will calibrate some of the most important and sensitive parameters of PnET-Succession. The calibration in the successive steps should be minimal if the instructions of step 1 have been followed, and if the initial parameters you are using are good at representing the differences between species.

## 📖 Format of this documentation

This documentation is a Jupyter Book, composed of several files (markdown files and Jupyter notebooks) that you can find in [the repository that contains the book](https://github.com/Klemet/Calibration_PnET_DIVERSE).

All of the actions used to calibrate PnET Succession as described in this book are thus executed in Python, with both the code and its outputs being visibile to you as you go through the book.

This means that you can replicate the whole process easily on your own computer. You can also download the files and scripts used throughout this book to use them for your own research, which should help you win some time.

➡️ **📖 If you simply want to inform yourself as to the calibration process, just keep reading !**

➡️ **💻 If you want to run the code used for this calibration process on your own computer (for example, to calibrate new species for your own study area), read the instructions of [the readme of the repository](https://github.com/Klemet/Calibration_PnET_DIVERSE/blob/main/README.md)**. They will help you deploy a Docker image on your computer that will contains the entire environment (including LANDIS-II) used for this calibration within minutes. 

**📖 Happy reading !**

## ✒️ Table of contents

```{tableofcontents}
```
