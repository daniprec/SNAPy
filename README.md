# SNAPy

## Set up
### Create the environment using Conda

  1. Install miniconda
     
     ```
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash
     ```

     Say yes to everything and accept default locations. Refresh bash shell with `bash -l`

  2. Update conda
     
      ```
      conda update -n base -c defaults conda
      ```

  3. Clone this repository and cd into the folder

  4. Create and activate conda environment (removing previously existing env of the same name)
     
       ```
       conda remove --name snapy --all
       conda env create -f environment.yml --force
       conda activate snapy
       ```

  5. (Optional) Install the Jupyter kernel
       ```
       pip install --user ipykernel
       python -m ipykernel install --user --name=snapy
       ```