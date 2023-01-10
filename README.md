# SPADE: State Space Augmented Transformer

This PyTorch package implements the language modeling experiments in [Efficient Long Sequence Modeling via State Space Augmented Transformer
](https://arxiv.org/abs/2212.08136).


## Dependencies

* The package runs on PyTorch v1.11.0 with CUDA 11.3. Note that installation of CUDA and cuDNN are required.
* Our implementation requires [fairseq](https://github.com/facebookresearch/fairseq) v0.11+.


## Installation

* Download [fairseq](https://github.com/facebookresearch/fairseq). Place `spade-modules/` in the downloaded directory.
  * If training of S4 is desirable (see instructions), replace `fairseq/trainer.py` with the provided `trainer.py` file.
    Specifically, we modified the `trainer.py/_build_optimizer()` function to add separate S4 parameter groups. 

* Run `pip install -e .` to install fairseq locally.

* Run `pip install -r requirements.txt` to install dependencies.

* Our implementation requires computing the Cauchy kernel.
  Note that `spade-modules/s4.py` largely depends on [this file](https://github.com/HazyResearch/state-spaces/blob/main/src/models/s4/s4.py).
  There are two possible ways:
  * (**Recommended**) Install [PyKeOps](https://www.kernel-operations.io/keops/python/index.html) using `pip install pykeops`.
  * Download the `extensions/` folder from [this repo](https://github.com/HazyResearch/state-spaces), and place the downloaded folder to `spade-modules/extensions/`. 
    Run `python setup.py install` within `spade-modules/extensions/cauchy/` to install the CUDA kernel.
    

## Instructions

* Follow the instructions [here](https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/README.md) to pre-process wikitext-103 data. 
  Run `bash run_lm.sh` to train a SPADE model.
  
* Empirically, we can initialize the S4 module and then freeze its parameters during training.
  We find that training the S4 parameters only provides marginal performance gain.
  
* Learning rates of S4 parameters are specified using the `--s4-lr` argument, 
  which is a dictionary containing three keys: `A`, `B` and `dt`. Each key can use three types of values:
  * `null` means the parameter uses the same learning rate as model parameters; 
  * `0.0` means the parameter is not trained;
  * `float` (>0.0) specifies a learning rate.
  

## Notes

### Contact Information

For personal communication related to this package, please contact Simiao Zuo (`simiaozuo@gatech.edu`), Xiaodong Liu (`xiaodl@microsoft.com`), or Jian Jiao (`jian.jiao@microsoft.com`).
  

### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


## Reference

Please cite the following paper if you use this package:
```
@article{zuo2022efficient,
  title={Efficient Long Sequence Modeling via State Space Augmented Transformer},
  author={Zuo, Simiao and Liu, Xiaodong and Jiao, Jian and Charles, Denis and Manavoglu, Eren and Zhao, Tuo and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2212.08136},
  year={2022}
}
```
