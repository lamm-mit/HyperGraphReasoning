#### Isabella Stewart, Markus J. Buehler, MIT, 2025 
Corresponding author: mbuehler@MIT.EDU

#### LAMM, Massachusetts Institute of Technology

## Summary

Modern scientific inquiry increasingly demands systems-level reasoning that unifies heterogeneous data with mechanistic understanding. Although large language models show promise for supporting such reasoning, they rely on structured knowledge representations such as knowledge graphs (KGs) to perform such tasks. However, conventional pairwise knowledge graphs are ill-suited to represent the complex, higher-order relationships common in scientific domains. In this work, we present a hypergraph-based framework for constructing knowledge representations from large-scale scientific text corpora, allowing explicit modeling of multi-entity interactions. Applied to a corpus on biocomposite scaffolds, hypergraph analysis uncovers a scale-free knowledge architecture characterized by highly connected conceptual hubs, while elucidating co-occurring concepts and their interdependencies. We further enable agentic reasoning by equipping LLM-based agents with the ability to traverse the hypergraph, demonstrating that this knowledge substrate grounds reasoning and facilitates mechanistic inference for hypothesis generation. Overall, our findings highlight the potential of hypergraph-based knowledge structures to enhance downstream scientific discovery from large and complex datasets.

Our contributions are:
1. A method for constructing hypergraphs from large-scale scientific text.
2. Techniques for analyzing hypergraphs derived from scientific corpora.
3. Agentic reasoning tools that operate over hypergraph representations.


## Getting Started

## 1) Instantiate Environment Configuration:

```
conda create -n LLM_hypergraphreasoning python=3.11 -y
conda activate LLM_hypergraphreasoning
```

## 2) Install CUDA (Recommended via Conda) 

Note: Ensure that the NVIDIA CUDA driver is already installed on your system before proceeding.

```
conda install -c "nvidia/label/cuda-12.6.0" cuda-toolkit cuda-nvcc -y --copy
```

## 3) (Optional) Install llama-ccp-python
Required only if you intend to run models locally (≥ v0.2.0). 
You can skip this step if you’re using API-based LLMs.

```
CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=on" FORCE_CMAKE=1 pip install git+https://github.com/abetlen/llama-cpp-python@v0.3.8 --verbose
```


## 4) Install HyperGraphReasoning package
Clone the repository and proceed with the direct installation.
```
git clone https://github.com/lamm-mit/HyperGraphReasoning.git
cd HyperGraphReasoning
pip install .
```

See requirements.txt for the full list of packages seen in the approach described in our Methods section (install packages as necessary) 

***

![Project Logo](agents.png)

***
# Generating a Hypergraph

## 1) PDF2Markdown
To convert PDF to markdown, follow instructions for creating environment at https://github.com/datalab-to/marker.git

It's best to make a clean environment for marker separate from our LLM_hypergraphreasoning environment just for this markdown conversion. 

```bash
conda activate marker
```

To convert multiple files at once: 
```bash
marker /path/to/input/folder
```
To convert multiple files on multiple GPUs: 
```bash
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
```

Final Composite Paper List.xlsx contains the list of papers from the corpus on 'biocomposite scaffold' that were used in this study. We do not provide the full text PDFs nor the markdown files in this repository. 


## 2) Generating Hypergraph and Hypergraph Embeddings from Markdown Files

Convert the Jupyter notebook to a Python script:
```bash
jupyter nbconvert --to script make_hypergraph.ipynb
```
This will create 

```bash
make_hypergraph.py
```

Run the generated script: 
```bash
python make_hypergraph.py
```
(optional) Submit the job script (e.g., SLURM):

```bash
sbatch run_make_hypergraph.sh
```

While any model can be used, `make_hypergraph.ipynb` supports the approach described in our Methods section by using the Together API to provide hosted LLM inference with `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` for graph generation.


## 3) Run Hypergraph Agents 
Open and run the following notebook in **JupyterLab**:
```bash
Agents.ipynb
```
The agents use: 

Generated hypergraph (compatible with HYPERNETX) can be found in
`Notebooks/SG/GRAPHDATA_OUTPUT_paper/final_graph.pkl`

Generated hypergraph with all metadata can be found in
`Notebooks/SG/GRAPHDATA_OUTPUT_paper/updated_sub_dfs.pkl`

Generated hypergraph embeddings can be found in
`Notebooks/SG/GRAPHDATA_paper/composite_LLAMA4_70b.pkl`

While any model can be used, `Agents.ipynb` supports the approach described in our Methods section by using the locally hosted LLM model `meta-llama/Llama-3.3-70B-Instruct` for inference. Our embedding model `nomic-ai/nomic-embed-text-v1.5` is also loaded locally using the `SentenceTransformers` library.

## 4) Analyze Hypergraph
Open and run the following notebook in **JupyterLab**:
```bash
analyze_hypergraph.ipynb
```

The analysis uses: 
Generated hypergraph (compatible with HYPERNETX) can be found in
`Notebooks/SG/GRAPHDATA_OUTPUT_paper/final_graph.pkl`

***


## Citation

Please cite this work as:

```bibtex
@article{stewart2025hypergraphreasoning,
  title     = {HIGHER-ORDER KNOWLEDGE REPRESENTATIONS FOR AGENTIC
SCIENTIFIC REASONING},
  author    = {x},
  journal   = {x},
  year      = {2025},
  doi       = {x}
}
