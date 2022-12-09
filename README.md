# Handwritten-Math-Expression-Recognition
Project for comp 473 

1. Download the required dependencies 
   - python -m pip install -r requirements.txt (if you use pip and venv)
   - conda install --file requirements.txt (if you use conda)
   - there is some issues downloading pytorch from the requirements file, please follow the instructions on: https://pytorch.org/get-started/locally/

2. Download the data
   - Go to https://tc11.cvc.uab.es/datasets/ICFHR-CROHME-2016_1
   - Create an account and download the zip file
   - You only need the following two folders from CROHME2016_data:
      - Task-1-Formula
      - Task-2-Symbols
   - You need this folder from CROHME2013_DATA:
      - TrainINKML
   - Unzip all the files (can take a while)
   - Put them in your local data repository
   
3. Convert the data from inkml to png
   - To convert all the files from a directory and its subdirectories, use : src/data_processing/convertfiles.py
   - To convert files from one directory or one file, use: src/data_processing/convertInkmlToImg.py
   
  
4. Train the model
   - run trainingdemoSymbols.ipynb
   - if you want to train the pre-trained model, uncomment one of the cell in trainingDemoSymbols.ipynb
   
5. Testing the model
   - run the cells in testingDemoSymbols.ipynb
   
   
