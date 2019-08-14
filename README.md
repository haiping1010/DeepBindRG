# DeepBindRG
A DEEP LEARNING BASED METHOD FOR ESTIMATING EFFECTIVE PROTEIN-LIGAND AFFINITY




prerequirment:

anaconda python 2.7
tensorflow
keras
scklearn
numpy
pandas



##################
#################
For testing cases that have experimental value
step 1:
unzip the all_data/data.zip, by cd all_data; unzip data.zip

step 2:
got the performance estimator(r value, rmse, etc) by:
python deep_learn_rob_residual_zhpxxx_n_regression_load_drop50.py


step 3:
list the vina score, the DeepBindRG prediction value, and the experimental value
(all_energies.sort is the output file from vina docking)
python perform_vina.py

check the output file  out_list.csv

collumn 1-4 are :
name,experiment value, DeepBindRG value, Vina value

############################
############################
For application cases that have no experimental value:

python deep_learn_rob_residual_zhpxxx_n_regression_load_drop50_use.py

check the out_file.csv




Citation：
DeepBindRG: a deep learning based method for estimating effective protein-ligand affinity


If there is any technique problem, please no hestitated to contact by email hp.zhang@siat.ac.cn.

 

