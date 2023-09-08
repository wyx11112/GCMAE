<p>
  <img src="img/framework.pdf" width="1000">
  <br />
</p>
The overview of GCMAE.
<hr>

<h1> Generative and Contrastive Paradigms Are Complementary for Graph Self-Supervised Learning </h1>

GCMAE is a self-supervised graph representation method, which unfies the contrastive learning and graph masked autoencoder. We conducted extensive experiments on various graph tasks, including *node classification*, *link prediction*, *node clustering*, and *graph classification*.


<h2>Dependencies </h2>

* Python >= 3.7
* [Pytorch](https://pytorch.org/) >= 1.9.0 
* [dgl](https://www.dgl.ai/) >= 0.7.2
* pyyaml == 5.4.1
* munkres


<h2>Quick Start </h2>

For quick start, you could run the scripts: 

**Node classification**

```bash
# Run the code manually for node classification:
python main.py --dataset cora --device 0
```

**Link prediction**

```bash
# Run the code manually for link prediction:
python main_lp.py --dataset cora --device 0 
```

**Node clustering**

```bash
# Run the code manually for node clustering:
python main.py --dataset cora --task cls --device 0 
```

**Graph classification**

```bash
# Run the code manually for graph classification:
python main_graph.py --dataset IMDB-BINARY --device 0 
```

Run  with `--use_cfg` in command to reproduce the reported results.

