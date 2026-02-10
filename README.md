<div align="center">
	<img align="middle" src="src/podlogo.png" width="400" alt="logo"/>
  <h2> Materials discovery acceleration by using conditional generative methodology</h2> 
</div>

paper: [Materials discovery acceleration by using conditional generative methodology](https://www.nature.com/articles/s41524-025-01930-w)

arXiv: [https://arxiv.org/abs/2505.00076](https://arxiv.org/abs/2505.00076)

data: [https://in.iphy.ac.cn/eln/link.html#/113/G9f5](https://in.iphy.ac.cn/eln/link.html#/113/G9f5)


## Environment

We recommend using Anaconda to manage Python environments. First, create and activate a new Python environment:
```
conda create --name podgen310 python=3.10
conda activate podgen310
```

Then, use `requirements.txt` to install the Python packages.
```
pip install -r requirements.txt
```

Finally, the PyTorch-related libraries need to be installed according to your device and CUDA version. The version we used is:
```
torch                    2.3.0+cu118
torchaudio               2.3.0+cu118
torchvision              0.18.0+cu118

torch_geometric          2.5.3
torch_cluster            1.6.3+pt23cu118
torch_scatter            2.1.2+pt23cu118
torch_sparse             0.6.18+pt23cu118
torch_spline_conv        1.2.2+pt23cu118

pytorch-lightning        2.4.0
torchmetrics             1.6.3
```
For details, you can refer to [PyTorch](https://pytorch.org), [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/#), [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/).

## How to run
After setting up the environment, you can use the provided model checkpoint to run PODGen for conditional generation of topological materials. Before doing so, make sure to update the necessary environment paths. You can either run the following commands:

```
cp .env_bak .env
bash writeenv.sh
```

Or, if you prefer, modify the .env file manually. Update it with the following lines, replacing <YOUR_PATH_TO_PODGEN> with the absolute path to your PODGen directory:


```
export PROJECT_ROOT="<YOUR_PATH_TO_PODGEN>/PODGen"
export HYDRA_JOBS="<YOUR_PATH_TO_PODGEN>/PODGen/output/hydra"
export WABDB_DIR="<YOUR_PATH_TO_PODGEN>/PODGen/output/wandb"
```

Download the model checkpoint from [this link](https://in.iphy.ac.cn/eln/link.html#/113/G9f5), and save the file ```epoch=645-step=1881798.ckpt``` to the ```src/models/Alex``` directory.


Then you can run the following command to generate crystal structures:
```
python podgen/mcmc_gen.py --config <YOUR_PATH_TO_PODGEN>/PODGen/conf/gen/MCMC_config.yaml
```

Of course, you can also try modifying other parameters in `conf/gen/MCMC_config.yaml`. The specific meaning of each parameter is explained within the file as well.


## Training new models

### Generative model
#### Train with mp20
We have provided the mp20 dataset, which was collected by [Tian Xie et al](https://github.com/txie-93/cdvae/tree/main/data/mp_20). You can try to use this dataset to train the model with th following command:
```
python CFtorch/run.py data=mp20 expname=mp20 
```
If you want to train this model with Multiple GPUs using 'DDP' strategy, you can try this command:
```
torchrun --nproc_per_node 2 CFtorch/run.py \
        data=mp20 \
        data.use_exit=True \
        expname=mp20 \
        train.pl_trainer.devices=2 \
        train.pl_trainer.strategy=ddp \
        train.pl_trainer.accelerator=gpu \
```
The values of “nproc_per_node” and “train.pl_trainer.devices” should be equal and set to the number of GPUs to be used.

#### Train with other dataset
If you have prepared your own dataset, you can organize it in the same format as `data/mp_20`. For training the generative model, make sure that each CSV file contains a column named "cif". Then, create a `<YOUR_DATA>.yaml` file under `conf/data/`, referring to the example `conf/data/mp20.yaml`. After that, specify `data=<YOUR_DATA>` in the model training command.

### Predictive model
#### Train with mp20
You can try to use mp20 to train the model with th following command:
```
python predictor/run.py data=pre_mp20 expname=premp20
```
If you want to train this model with Multiple GPUs using 'DDP' strategy, you can try this command:
```
torchrun --nproc_per_node 2 predictor/run.py data=pre_mp20 expname=premp20 accelerator=ddp
```
The values of “nproc_per_node” set to the number of GPUs to be used.

By setting the variables "prop", "use_prop", and "num_targets" in file `conf/data/pre_mp20`, you can control which property the prediction model is used for and whether it is a regression or classification model. If "num_targets" is set to an integer greater than 1, the trained model will be a classification model. Of course, you can also modify these parameters directly from the command line. For example, you can set:
```
python predictor/run.py data=pre_mp20 expname=premp20 data.use_prop=band_gap data.num_targets=1
```

#### Train with other dataset
If you have prepared your own dataset, you can organize it in the same format as `data/mp_20`. For training the predictive model, make sure that each CSV file contains a column named "cif" and a column that records the property you want to use. Then, create a `<YOUR_DATA>.yaml` file under `conf/data/`, referring to the example `conf/data/pre_mp20.yaml`. After that, specify `data=<YOUR_DATA>` in the model training command. Pay special attention to the settings of the three parameters: "prop", "use_prop", and "num_targets".
