<p align="center"><img src="./DIVERSE_PnET_Calibration_Logo.png" width="220"></p>

<h1 align="center">Calibration of PnET Succession for the <a src="https://diverseproject.uqo.ca">DIVERSE project</a></h1>

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://klemet.github.io/Calibration_PnET_DIVERSE/)

## 🎯 Objective

- This repo contains the files used to calibrate the canadian tree species we will simulate with the [PnET Succession Extension](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession) of the [LANDIS-II Forest Landscape Model](https://www.landis-ii.org/).
- These parameters will be used in simulations necessary for the [research theme 4](https://diverseproject.uqo.ca/theme-4-evaluation-various-forest-management-approaches/) of the pan-canadian [DIVERSE research project](https://diverseproject.uqo.ca), where several Functional Management Unit of Canada will be simulated with different forest management strategies.
- The calibration process used here is mostly based on the [calibration tips](https://research.fs.usda.gov/about/people/gustafson) of Eric Gustafson, available in the [PnET Succession v5.1 user guide](https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession/blob/master/deploy/docs/LANDIS-II%20PnET-Succession%20v5.1%20User%20Guide.pdf).
    - The calibration is done in 5 steps of increasing complexity : calibrating species in monocultures and ideal conditions; calibrating species competition for light; calibration species competition for water; calibration of species competition with changing temperatures; and finally calibrating species establishment in the landscape.
- **💡 One of the main objectives here is to make this calibration process transparent and completly replicable, so as to facilitate future studies that will want to use PnET in different contexts, or with future versions**.
    - See the next section ⚙ **How to use** in order to deploy all of the files and programs used for this calibration on your computer. The process should be relatively simple. Feel free to post an issue if you encouter a problem.

## 💻 How to run it on your computer

- Install Docker Desktop on your computer (https://www.docker.com/products/docker-desktop/) or simply install Docker if you are using Linux.
- Clone or download the contents of this repository on your computer.
- Go to the folder `Clean_Docker_LANDIS-II_8_AllExtensions_PnETCalibration` in a terminal where you can call Docker (e.g. Windows Powershell)
- Use `docker build -t landis_ii_v8_calibration_pnet ./` to build the docker image. This image contains everything needed to re-do the calibration I've done here in a replicable way (including Jupyter lab to see read and act on the Jupyter notebooks file which have been used to create the Jupyter Book available [here](https://klemet.github.io/Calibration_PnET_DIVERSE/)), and of course, LANDIS-II. The `Dockerfile` file contain all of the commands used to build the image, and comments about what versions of the LANDIS-II extensions are installed.
> ⚠️ You might get some errors during the build. These are often due to some downloads from github failing for a reason unkown. Just re-launch the build process, and everything should be fine. If things are still not working, please [post an issue](https://github.com/Klemet/Calibration_PnET_DIVERSE/issues) with the error message that you are getting.
- Once the build is over, run `docker run -it -p 8888:8888 --mount type=bind,src="<CALIBRATION_FOLDER_PATH>",dst=/calibrationFolder landis_ii_v8_calibration_pnet` in a terminal to open an interactive session with docker. Replace `<CALIBRATION_FOLDER_PATH>` with the full path to where the folder `Calibration_PnET_DIVERSE` is on your computer. It is advised that you keep the bind to `/calibrationFolder` in the container, as Jupyter lab is configured to look there for its configuration files.
> ⚠️ **If you close the terminal where you have entered the `docker run` command, the Docker container will close down with it !**
- Jupyter lab will automatically launch when launching the container. Simply click on the URL that it gives you (that will look like http://127.0.0.1:8888/lab?token=<LONG TOKEN>) or copy/paste it in your favorite web browser.
- In Jupyter lab, use the navigation pannel on the left to go to `/calibrationFolder` to open the Jupyter notebooks corresponding to the part of the calibration you are interested in. You can normally run every cell without any compatibility issue from here.
- If you are using these files for your own calibration process, and if you need to add/update Python packages to the docker image to use them in the notebook, you can quit the jupyter lab instance and use `pip` to install them. However, these will be temporary. Editing the `Dockerfile` to add the `pip install ...` commands you need at the end and then re-building the image will make things permanent.

## 🛠️ Technical notes about using the docker image and building the jupyter book used in this repository

- When building the Docker image used to run everything in this repository, the program `pip` is used to install the packages necessary for the notebooks to execute. In order to make the notebook future-proof and avoid a situation where commands are not working anymore because of package updates, I first installed all of the packages while letting pip choose the versions it wants for each (to resolve the dependancies between packages); then, testing if this combination of packages works with the notebook; then, exporting the specific versions for each package with the command `pip freeze` in a text file. This text file is then used to build the Docker image (`pip_requirements.txt`).
    - As such, **f you need to install new packages**, you might need to edit the Dockerfile to use `pip_requirements_NotFreezed.txt` where the versions are not specified so that pip can resolve the dependancies of your new package combined with the others. Just replace `pip_requirements.txt` by `pip_requirements_NotFreezed.txt` in the Dockerfile and add your new package to `pip_requirements_NotFreezed.txt`. **If you don't do this, pip might become unable to resolve the package dependencies when building the Docker image, and you might get an error**.
- Don't try to stage files with git or create git commit while you are running the Jupyter lab server on your computer. **This will likely results in errors in git where git will become unable to find some of the files in your folders, even though they exist**.
- Use `BuildAndUpdateJupyterBook.ipynb` in Jupyter lab to re-create the Jupyter Book, whose file will be put into `/docs`. Pushing a commit with new files in `/docs` will re-create the github page.
