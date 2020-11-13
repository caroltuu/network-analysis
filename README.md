# Network analysis project

## Instructions

Train models:
```
python3 main.py --filename trial --n_models 10 --train              # on cpu
python3 main.py --filename trial --n_models 10 --train -- gpu 0     # on GPU 0
```
--filename flag used when saving models (base name of the file to save)
--n_models flag states how many models to train simultanously
--train flag traines all models
--gpu flag allows to choose between gpu and cpu training

To remove past results:
```
python3 main.py --clear
```
--clear flag removes past results

To visualize current results:
```
python3 main.py --filename trial --n_models 10 --visualize
```
--visualize flag runs the visualizations on the results