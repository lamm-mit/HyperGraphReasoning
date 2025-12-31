from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *

from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import networkx as nx
import os
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
from pyvis.network import Network
from tqdm.notebook import tqdm

import seaborn as sns

from hashlib import md5


#hypergraph add ons
import json #do we need? 
import re #do we need? 
import hypernetx as hnx
import pickle



palette = "hls"
# Code based on: https://github.com/rahulnyk/knowledge_graph


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
            "chunk_id": md5(chunk.encode()).hexdigest(),
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)

    return df


def df2Graph(df: pd.DataFrame, generate, generate_figure=None, image_list=None, repeat_refine=0, do_distill=True, do_relabel = False, verbatim=False,
          
            ) -> nx.DiGraph:
    
    subgraph_list = []
    for _, row in df.iterrows():
        subgraph = graphPrompt(
            row.text, 
            generate,
            generate_figure, 
            image_list,
            {"chunk_id": row.chunk_id}, 
            do_distill=do_distill,
            do_relabel=do_relabel,
            repeat_refine=repeat_refine, 
            verbatim=verbatim,
        )
        print(subgraph, type(subgraph))
        subgraph_list.append(subgraph)

        
    G = nx.DiGraph()

    for g in subgraph_list:
        G = nx.compose(G, g)
    
    return G

def df2hypergraph(
    df: pd.DataFrame,
    generate,
    generate_figure=None,
    image_list=None,
    repeat_refine=0,
    do_distill=True,
    do_relabel=False,
    verbatim=False,
) -> hnx.Hypergraph:
    """
    Build one big HyperNetX hypergraph by unioning together
    all of the sub-hypergraphs produced for each row of the DataFrame.
    """
    sub_hgs = []
    sub_dfs = []

    for _, row in df.iterrows():
        try:
            hg, chunk_df = hypergraphPrompt(
                row.text,
                generate,
                generate_figure,
                image_list,
                {"chunk_id": row.chunk_id},
                do_distill=do_distill,
                do_relabel=do_relabel,
                repeat_refine=repeat_refine,
                verbatim=verbatim,
            )
            # Only keep valid subgraphs
            if isinstance(hg, hnx.Hypergraph):
                sub_hgs.append(hg)
                sub_dfs.append(chunk_df)
            else:
                print(f"Skipping chunk {row.chunk_id} – returned non-Hypergraph")
        except Exception as e:
            print(f"Exception while processing chunk {row.chunk_id}: {e}")

    if not sub_hgs:
        print("No valid subgraphs found. Returning None.")
        return None, None

    # Start from an empty hypergraph
    H = hnx.Hypergraph({})

    # Union them all safely
    for hg in sub_hgs:
        H = H.union(hg)

    return H, sub_dfs

import sys
sys.path.append("..")

import json

def graphPrompt(input: str, generate, generate_figure=None, image_list=None, metadata={}, #model="mistral-openorca:latest",
                do_distill=True, repeat_refine=0,verbatim=False,
               ) -> nx.DiGraph:
    
    try:
        return nx.read_graphml(f"temp/{metadata['chunk_id']}.graphml")
    except:
        pass

    os.makedirs('temp', exist_ok = True) 

    SYS_PROMPT_DISTILL = f'You are provided with a context chunk (delimited by ```) Your task is to respond with a concise scientific heading, summary, and a bullited list to your best understaninding and all of them should include reasoning. You should ignore human-names, references, or citations.'
    
    USER_PROMPT_DISTILL = f'In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own and provide all background needed, and include details. Ignore references. Extract the table if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any.'
        
    SYS_PROMPT_FIGURE = f'You are provided a figure that contains important information. Your task is to analyze the figure very detailedly and report the scientific facts in this figure. If this figure is not an academic figure you should return "". Always return the full image location.'
    
    USER_PROMPT_FIGURE = f'In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own and provide all background needed, and include details. Extract the image if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any.'
    input_fig = ''
    if generate_figure: # if image in the chunk
        
        for image_name in image_list:
            _image_name = image_name.split('/')[-1]
            if _image_name.lower() in input.lower():  
                input_fig =  f'Here is the information in the image: {image_name}' + \
                generate_figure( image = image_name, system_prompt=SYS_PROMPT_FIGURE, prompt=USER_PROMPT_FIGURE)
    
    if do_distill:
        #Do not include names, figures, plots or citations in your response, only facts."
        input = generate( system_prompt=SYS_PROMPT_DISTILL, prompt=USER_PROMPT_DISTILL)

    if input_fig:
        input += input_fig
    
    SYS_PROMPT_GRAPHMAKER = (
        'You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. '
        'You are provided with a context chunk (delimited by ```) Your task is to extract the ontology of terms mentioned in the given context, representing the key concepts as per the context with well-defined and widely used names of materials, systems, methods.'
        'You always report a technical term or abbreviation and keep it as it is.'
        'If you receive a location to an image, you must use it as a node which <id> will be the location and the <type> will be "image" and relate the information in the context to make the nodes and edges relation.'
        '<relation> in an edge must truly reveal important information that can provide scientific insight from the <source> to the <target>'
        'Return a JSON with two fields: <nodes> and <edges>.\n'
        'Each node must have <id> and <type>.\n'
        'Each edge must have <source>, <target>, and <relation>.'
    )
     
    USER_PROMPT = f'Context: ```{input}``` \n\ Extract the knowledge graph in structured JSON: '
    # result = [dict(item, **metadata) for item in result]
    
    print ('Generating triples...')
    result  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)

    G = nx.DiGraph()
    for node in result.nodes:
        G.add_node(node.id, type=node.type)
    for edge in result.edges:
        G.add_edge(edge.source, edge.target, relation=edge.relation, chunk_id=metadata['chunk_id'])

    nx.write_graphml(G, f"temp/{metadata['chunk_id']}.graphml")
    print(f'Generated graph: {G}')

    return G


def hypergraphPrompt(input: str, generate, generate_figure=None, image_list=None, metadata={}, #model="mistral-openorca:latest",
                do_distill=True, do_relabel=False,repeat_refine=0,verbatim=False,
               ) -> hnx.Hypergraph:
    
    try:
        with open(f"temp/{metadata['chunk_id']}.pkl", "rb") as fin:
            H = pickle.load(fin)
        if verbatim:
            print(f"Loaded hypergraph from temp/{metadata['chunk_id']}.pkl")
        return H
    except:
        pass

    # Ensure cache directory exists
    os.makedirs('temp', exist_ok=True)
    SYS_PROMPT_DISTILL = f'You are provided with a context chunk (delimited by ```) Your task is to respond with a concise scientific heading, summary, and a bullited list to your best understaninding and all of them should include reasoning. You should ignore human-names, references, or citations.'
    
    USER_PROMPT_DISTILL = f'In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own and provide all background needed, and include details. Ignore references. Extract the table if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any.'
        
    SYS_PROMPT_FIGURE = f'You are provided a figure that contains important information. Your task is to analyze the figure very detailedly and report the scientific facts in this figure. If this figure is not an academic figure you should return "". Always return the full image location.'
    
    USER_PROMPT_FIGURE = f'In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own and provide all background needed, and include details. Extract the image if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any.'
    input_fig = ''
    if generate_figure: # if image in the chunk
        
        for image_name in image_list:
            _image_name = image_name.split('/')[-1]
            if _image_name.lower() in input.lower():  
                input_fig =  f'Here is the information in the image: {image_name}' + \
                generate_figure( image = image_name, system_prompt=SYS_PROMPT_FIGURE, prompt=USER_PROMPT_FIGURE)
    
    if do_distill:
        #Do not include names, figures, plots or citations in your response, only facts."
        input = generate( system_prompt=SYS_PROMPT_DISTILL, prompt=USER_PROMPT_DISTILL)

    if input_fig:
        input += input_fig

    SYS_PROMPT_GRAPHMAKER = (
    "You are a network ontology graph maker who extracts precise Subject–Verb–Object triples from a given context.\n\n"
    "You are provided with a context chunk (delimited by triple backticks: ```).\n"
    "Produce two passes:\n"
    "  1) First pass: exact grammatical S–V–O extraction (with composite detection).\n"
    "  2) Second pass: conservative semantic completion for relations that are present in raw text but not realized as clean S–V–O.\n\n"
    "Proceed step by step for EACH SENTENCE in the chunk:\n\n"
    "=== Composite Detection (pre-pass) ===\n"
    "Detect composite noun phrases joining multiple ingredients via '/', '-', 'and', '&', or commas\n"
    "(e.g., 'chitosan/hydroxyapatite nanocomposite rods', 'silk and collagen').\n"
    "Emit: {\"source\":[...], \"relation\":\"compose\", \"target\":\"<composite phrase>\"}.\n\n"
    "== First Pass: Exact S–V–O Extraction ==\n"
    "Thought 1.1: Split into sentences. For each sentence:\n"
    "  a) Identify grammatical Subject (S), Verb/Predicate (V), and Object/Complement (O).\n"
    "  b) Split multi-element S or O (other than composites) into lists, preserving order.\n"
    "  c) Copy S, V, O exactly as written (no paraphrase).\n"
    "  d) Keep only triples where V directly governs O or is a valid predicate:\n"
    "     • Transitives (e.g., 'exhibits', 'encapsulates', 'enhances', 'limits').\n"
    "     • Copular/predicative ('is/are/was/were', 'serve as', 'emerged as').\n"
    "     • Prepositional predicates with verbatim preposition ('used in/for/as', 'employed in', 'limited to', 'leads to', 'results in').\n"
    "  e) Relative clauses: if S or O has 'that/which … VERB …', emit an event with the head noun as subject\n"
    "     (e.g., {source:'nHAp', relation:'is employed in', target:'dentistry'}).\n"
    "  f) Emit one event per unambiguous predicate.\n\n"
    "== Second Pass: Conservative Semantic Completion (same sentence/clauses only) ==\n"
    "Goal: recover relations that the sentence clearly encodes but which lack a clean S–V–O surface form.\n"
    "Apply ONLY within the same sentence (or clearly attached clause). Do not cross sentences.\n"
    "For any terms not linked in Pass 1, consider these patterns:\n"
    "  1) Nominalizations → light verbs (relation only):\n"
    "     If a head noun clearly denotes an action or property, verbalize the relation while keeping nodes verbatim.\n"
    "     Examples: 'fabrication of X' → {source:'<agent/context if stated or omit>', relation:'fabricate', target:'X'};\n"
    "               'investigation of X' → 'investigate'; 'properties: porosity, biodegradability' → 'has'.\n"
    "  2) Apposition/definition:\n"
    "     'Collagen, a structural protein' → {source:'Collagen', relation:'is', target:'a structural protein'}.\n"
    "  3) Purpose / function / use phrases:\n"
    "     'X for Y', 'X to Y' (infinitival purpose), 'X intended for Y' → relation 'used for' or verbatim ('intended for').\n"
    "  4) Causal/resultive connectives not realized as main verb:\n"
    "     'Because/thereby/hence/therefore …' → 'leads to' / 'results in' if the cause–effect is explicit.\n"
    "  5) Prepositional attributions outside main predicate:\n"
    "     Headings/lists like 'Scaffold properties: porosity, biodegradability' → {source:'Scaffold', relation:'has', target:'porosity'}, etc.\n"
    "Constraints for Pass 2:\n"
    "  • Keep sources/targets verbatim from the sentence; only the relation may be abstracted.\n"
    "  • At most ONE inferred relation per unlinked term/group per sentence.\n"
    "  • Prefer specific verbalizations ('used for', 'results in') over generic 'is/has' when the phrase provides it.\n"
    "  • Never contradict Pass 1; do not merge distinct events.\n\n"
    "Important:\n"
    "\"Term specificity (resolve vague mentions):\""
    "\"- Avoid vague nodes like: \\\"material(s)\\\", \\\"material formulation\\\", \\\"formulation\\\", \\\"solution\\\", \\\"sample\\\", \\\"device\\\",\""
    "\"  \\\"method(s)\\\", \\\"technique(s)\\\", \\\"approach\\\", \\\"process\\\", \\\"system(s)\\\", \\\"structure(s)\\\", \\\"polymer\\\" (alone), \\\"composite\\\" (alone), \\\"matrix\\\" (alone), \\\"property/properties\\\" (alone).\""
    "\"- If a sentence contains a vague mention (e.g., \\\"this material\\\", \\\"the formulation\\\"), resolve it to the most specific noun phrase in the local context:\""
    "\"  1) Prefer the closest antecedent in the SAME sentence (apposition or earlier NP with modifiers).\""
    "\"  2) Otherwise, look back up to 1–2 sentences for the last specific NP that the demonstrative/pronoun refers to.\""
    "\"  3) Use exact surface form from the text (preserve modifiers: e.g., \\\"nHAp-based polymer nanocomposite scaffold\\\").\""
    "\"- Only emit the event if you can resolve the vague mention to a specific NP. If not resolvable, omit the event.\""
    "- Pass 1 must be verbatim predicates (include prepositions).\n"
    "- Preserve technical terms/modifiers in nodes; ignore human names/citations.\n"
    "- Only include fields 'source', 'relation', 'target'.\n\n"
    "- DO NOT make any events related to authors or investigators of the paper - as a guiding principle NO NAMES of people should be sources or targets\n"
    "- If you encounter a null byte or other non-printable control character in text, interpret it as a placeholder for the degree symbol (°) if followed by C or F, and output it in words (e.g., \"50 degrees C\").\n"
    "Otherwise, replace the character with its intended meaning based on context.\n"
    "Output Specification:\n"
    "Return a JSON object with a single field 'events' (a list of objects).\n"
    "Each object must have:\n"
    "- 'source': a list of strings (always use a list, even for a single source)\n"
    "- 'relation': a string (verbatim in Pass 1; abstracted in Pass 2)\n"
    "- 'target': a list of strings (always use a list, even for a single item)\n"
    "- All values must be flat (no nested lists or objects).\n\n"
    "Example output:\n"
    "{\n"
    "  \"events\": [\n"
    "    {\"source\": [\"chitosan\", \"hydroxyapatite\"], \"relation\": \"compose\", \"target\": [\"chitosan/hydroxyapatite nanocomposite rods\"]},\n"
    "    {\"source\": [\"nHAp-based materials\"], \"relation\": \"exhibit\", \"target\": [\"bioactive\", \"biocompatible\", \"osteoconductive features\"]},\n"
    "    {\"source\": [\"synthetic inorganic biomaterials\"], \"relation\": \"serve as\", \"target\": [\"an efficient and pathogen-free choice\"]},\n"
    "    {\"source\": [\"nHAp\"], \"relation\": \"is employed in\", \"target\": [\"dentistry\"]},\n"
    "    {\"source\": [\"nHAp\"], \"relation\": \"mimics\", \"target\": [\"the natural mineral composition of bones and teeth\"]},\n"
    "    {\"source\": [\"chitosan\", \"hydroxyapatite\"], \"relation\": \"form\", \"target\": [\"nanocomposite rod\"]},\n"
    "    {\"source\": [\"hydrogel matrix\"], \"relation\": \"encapsulates\", \"target\": [\"growth factor\", \"stem cells\", \"bioactive molecules\"]},\n"
    "    {\"source\": [\"the brittle nature of synthetic nHAp\"], \"relation\": \"leads to\", \"target\": [\"weak mechanical properties\"]},\n"
    "    {\"source\": [\"researchers\"], \"relation\": \"investigate\", \"target\": [\"nHAp-based polymer nanocomposite scaffolds for bone regeneration\"]},\n"
    "    {\"source\": [\"scaffold\"], \"relation\": \"has\", \"target\": [\"porosity\", \"biodegradability\"]},\n"
    "    {\"source\": [\"HAp-based composites\"], \"relation\": \"used for\", \"target\": [\"bone tissue engineering\"]},\n"
    "    {\"source\": [\"collagen\"], \"relation\": \"is\", \"target\": [\"a structural protein\"]}\n"
    "  ]\n"
    "}\n"
    "Return a JSON object in the exact format above. Remember no authors/names/investigators in events.\n" 
)
 
    #USER_PROMPT = f'Context: ```{input}``` \n\ Extract the hypergraph knowledge graph in structured JSON format: '
    USER_PROMPT = f'Context: ```{input}``` \n Extract the hypergraph knowledge graph in structured JSON format: '

    print ('Generating hypergraph...')
    validated_result  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)

    # 1) Build the raw edge→relation mapping
    edge_mapping = {
        f"e{i+1}": event.relation
        for i, event in enumerate(validated_result.events)
    }
    
    # 2) Build the base incidence dict
    base_edge_dict = {
        eid: (
            #set(event.source if isinstance(event.source, list) else [event.source])
            set(event.source) | set(event.target)
            #| {event.target}
        )
        for eid, event in zip(edge_mapping.keys(), validated_result.events)
    }
    
    # 3) Prepare the source/target/chunk maps
    source_map = {
        eid: event.source
        for eid, event in zip(edge_mapping.keys(), validated_result.events)
    }
    target_map = {
        eid: event.target
        for eid, event in zip(edge_mapping.keys(), validated_result.events)
    }
    chunk_map = {
        eid: metadata["chunk_id"]
        for eid in edge_mapping.keys()
    }
    
    # 4) Choose your edge IDs
    if do_relabel:
        # keep e1, e2, … IDs
        final_incidence = base_edge_dict
        final_source   = source_map
        final_target   = target_map
        final_chunk    = chunk_map

    else:
    # use human-readable relation names with unique suffixes
        renamed_edges = {
        eid: f"{edge_mapping[eid]}_chunk{chunk_map[eid]}_{i}"
        for i, eid in enumerate(edge_mapping.keys())
        }
    
        final_incidence = {
            renamed_edges[eid]: nodes
            for eid, nodes in base_edge_dict.items()
        }
        final_source = {
            renamed_edges[eid]: val
            for eid, val in source_map.items()
        }
        final_target = {
            renamed_edges[eid]: val
            for eid, val in target_map.items()
        }
        final_chunk = {
            renamed_edges[eid]: val
            for eid, val in chunk_map.items()
        }
        
    
    # 5) Create the HyperNetX hypergraph
    H_simple = hnx.Hypergraph(final_incidence)
    
    # 6) Build a combined DataFrame    
    rows = []
    for eid, nodes in final_incidence.items():
        rows.append({
            "edge":   eid,
            "nodes":  nodes,
            "source": final_source[eid],
            "target": final_target[eid],
            "chunk":  final_chunk[eid],
        })
    try:
        chunk_df = pd.DataFrame(rows).set_index("edge")
    except KeyError as e:
        print("Error during KG generation - skipping this chunk!:", e)
        return None, pd.DataFrame()
        
    print(
        f"Generated hypergraph with {len(H_simple.nodes)} nodes, "
        f"{len(H_simple.edges)} edges."
    )
    
    # 7) Return both graph and table
    return H_simple, chunk_df


def colors2Community(communities) -> pd.DataFrame:
    
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def make_graph_from_text (txt,generate, generate_figure=None, image_list=None,
                          graph_root='graph_root',
                          chunk_size=2500,chunk_overlap=0,do_distill=True, do_relabel=False,
                          repeat_refine=0,verbatim=False,
                          data_dir='./data_output_KG/',
                          save_HTML=False,
                          save_PDF=False,#TO DO
                         ):    
    
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
    graph_GraphML=  f'{data_dir}/{graph_root}.graphml'  #  f'{data_dir}/result.graphml',

    try:
        G = nx.read_graphml(graph_GraphML)
    except:

        outputdirectory = Path(f"./{data_dir}/") #where graphs are stored from graph2df function
        
    
        splitter = RecursiveCharacterTextSplitter(
            #chunk_size=5000, #1500,
            chunk_size=chunk_size, #1500,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        pages = splitter.split_text(txt)
        print("Number of chunks = ", len(pages))
        if verbatim:
            display(Markdown (pages[0]) )
        
        df = documents2Dataframe(pages)
        df.to_csv(f'{data_dir}/{graph_root}_chunks_clean.csv')

        G = df2Graph(df,generate, generate_figure, image_list, do_distill=do_distill, do_relabel=do_relabel, repeat_refine=repeat_refine,verbatim=verbatim) #model='zephyr:latest' )

        nx.write_graphml(G, graph_GraphML)
        

    graph_HTML = None
    net= None
    output_pdf = None
    if save_HTML:
        net = Network(
                notebook=True,
                cdn_resources="remote",
                height="900px",
                width="100%",
                select_menu=True,
                filter_menu=False,
            )

        net.from_nx(G)
        net.force_atlas_2based(central_gravity=0.015, gravity=-31)

        net.show_buttons()

        graph_HTML= f'{data_dir}/{graph_root}.html'
        net.save_graph(graph_HTML,
                )
        if verbatim:
            net.show(graph_HTML,
                )


        if save_PDF:
            output_pdf=f'{data_dir}/{graph_root}.pdf'
            pdfkit.from_file(graph_HTML,  output_pdf)
        
    
    return graph_HTML, graph_GraphML, G, net, output_pdf

def make_hypergraph_from_text(
    txt,
    generate,
    generate_figure=None,
    image_list=None,
    graph_root='graph_root',
    chunk_size=2500,
    chunk_overlap=0,
    do_distill=True,
    do_relabel=False,
    repeat_refine=0,
    verbatim=False,
    data_dir='./data_output_KG/',
):
    """
    Builds or loads a graph stored in a .pkl file.

    - If `{graph_root}.pkl` exists in `data_dir`, loads and returns it.
    - Otherwise, splits `txt` into chunks, generates a graph `G`, 
      pickles `G` to `{graph_root}.pkl`, and returns it.

    Returns:
    pkl_path (str), G 
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    pkl_path = os.path.join(data_dir, f'{graph_root}.pkl')
    sub_dfs_pkl_path = os.path.join(data_dir, f'original_sub_dfs.pkl')

    # Load or build the graph
    if os.path.isfile(pkl_path):
        with open(pkl_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded existing graph from {pkl_path}")
    else:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        pages = splitter.split_text(txt)
        print("Number of chunks =", len(pages))
        if verbatim:
            from IPython.display import Markdown, display
            display(Markdown(pages[0]))

        # Convert chunks to DataFrame
        df = documents2Dataframe(pages)
        df.to_csv(os.path.join(data_dir, f'{graph_root}_chunks_clean.csv'), index=False)

        # Generate graph
        G, sub_dfs = df2hypergraph(
            df,
            generate,
            generate_figure,
            image_list,
            do_distill=do_distill,
            do_relabel=do_relabel,
            repeat_refine=repeat_refine,
            verbatim=verbatim
        )

        # Save as pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Saved new graph to {pkl_path}")

        # Save sub_dfs as pickle
        with open(sub_dfs_pkl_path, 'wb') as f:
            pickle.dump(sub_dfs, f)
        print(f"Saved new graph to {sub_dfs_pkl_path}")
      

    return pkl_path, G, sub_dfs_pkl_path, sub_dfs


import time
from copy import deepcopy

def add_new_subgraph_from_text(txt=None,generate=None,generate_figure=None, image_list=None, 
                               node_embeddings=None,tokenizer=None, model=None, original_graph=None,
                               data_dir_output='./data_temp/',graph_root='graph_root',
                               chunk_size=10000,chunk_overlap=2000,
                               do_update_node_embeddings=True, do_distill=True, do_relabel = False, 
                               do_simplify_graph=True,size_threshold=10,
                               repeat_refine=0,similarity_threshold=0.95,
                               do_Louvain_on_new_graph=True, 
                               #whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,
                               graph_GraphML_to_add=None,
                               verbatim=True,):

    display (Markdown(txt[:32]+"..."))
    graph_GraphML=None
    G_new=None
    
    res=None
    # try:
    start_time = time.time() 

    if verbatim:
        print ("Now create or load new graph...")

    if (G_to_add is not None and graph_GraphML_to_add is not None):
        print("G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added.")
        return
    elif graph_GraphML_to_add==None and G_to_add==None: #make new if no existing one provided
        print ("Make new graph from text...")
        _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                 data_dir=data_dir_output,
                                 graph_root=f'graph_root',
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                 repeat_refine=repeat_refine, 
                                 verbatim=verbatim,
                                 )
        if verbatim:
            print ("New graph from text provided is generated and saved as: ", graph_GraphML_to_add)
    elif G_to_add is None:
        if verbatim:
            print ("Loading or using provided graph... Any txt data provided will be ignored...:", G_to_add, graph_GraphML_to_add)
            G_to_add = nx.read_graphml(graph_GraphML_to_add)
    # res_newgraph=graph_statistics_and_plots_for_large_graphs(G_to_add, data_dir=data_dir_output,                                      include_centrality=False,make_graph_plot=False,                               root='new_graph')
    print("--- %s seconds ---" % (time.time() - start_time))
    # except:
        # print ("ALERT: Graph generation failed...")
        
    print ("Now grow the existing graph...")
    
    # try:
    #Load original graph
    if type(original_graph) == str:
        G = nx.read_graphml(original_graph)
    else:
        G = deepcopy(original_graph)
    print(G, G_to_add)
    G_new = nx.compose(G, G_to_add)

    if do_update_node_embeddings:
        if verbatim:
            print ("Now update node embeddings")
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if do_simplify_graph:
        if verbatim:
            print ("Now simplify graph.")
        G_new, node_embeddings = simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                verbatim=verbatim,)
    if size_threshold >0:
        if verbatim:
            print ("Remove small fragments")            
        G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if return_only_giant_component:
        if verbatim:
            print ("Select only giant component...")   
        connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
        G_new = G_new.subgraph(connected_components[0]).copy()
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if do_Louvain_on_new_graph:
        G_new=graph_Louvain (G_new, graph_GraphML=None)
        if verbatim:
            print ("Done Louvain...")

    if verbatim:
        print ("Done update graph")

    graph_GraphML= f'{data_dir_output}/{graph_root}_integrated.graphml'
    if verbatim:
        print ("Save new graph as: ", graph_GraphML)

    nx.write_graphml(G_new, graph_GraphML)
    if verbatim:
        print ("Done saving new graph")
    
    # res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,make_graph_plot=False,root='assembled')
    # print ("Graph statistics: ", res)

    # except:
        # print ("Error adding new graph.")
    print(G_new, graph_GraphML)
        # print (end="")

    return graph_GraphML, G_new, G_to_add, node_embeddings, res


#START SANITIZE

import pandas as pd, re
import hypernetx as hnx

_CC = re.compile(r'[\x00-\x1F\x7F]')

def _scrub(s) -> str:
    s = "" if s is None else str(s)
    s = _CC.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _flatten_props_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(index=pd.Index([], name="edge"))
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        if "edge" in df.columns:
            df = df.set_index("edge")
        else:
            idx_cols = [c for c in df.columns if str(c).startswith("level_")]
            if not idx_cols:
                idx_cols = list(df.columns[:2])
            df["edge"] = df[idx_cols].astype(str).agg("|".join, axis=1)
            df = df.drop(columns=idx_cols).set_index("edge")
    df.index = pd.Index([_scrub(ix) for ix in df.index], name=df.index.name or "edge")
    return df

def _collect_incidence(G):
    """Return {edge_id: [node,...]} robustly across HNX versions (Hypergraph or HypergraphView)."""
    # 1) Best: incidence_dict (edge -> set(nodes))
    if hasattr(G, "incidence_dict"):
        return {e: [n for n in nodes] for e, nodes in G.incidence_dict.items()}

    # 2) Next: indexable edges (edges[e] -> iterable of nodes)
    try:
        return {e: list(G.edges[e]) for e in G.edges}
    except Exception:
        pass

    # 3) Fallback: rebuild from node→edges (adjacency_dict)
    if hasattr(G, "adjacency_dict"):
        inc = {}
        for n, edges in G.adjacency_dict.items():
            for e in edges:
                inc.setdefault(e, set()).add(n)
        return {e: list(ns) for e, ns in inc.items()}

    # If all else fails, raise with a hint
    raise TypeError("Could not extract incidence from HyperNetX object; version/API mismatch.")

def normalize_hnx_graph_rebuild(G: hnx.Hypergraph) -> hnx.Hypergraph:
    """
    Build a *new* Hypergraph with:
      - clean string edge ids (tuples -> 'a|b', control chars stripped)
      - scrubbed node labels
      - edge_properties single-index DF aligned to edge ids
    """
    # Edge id map (tuple -> joined; scrub)
    edge_map = {}
    for e in G.edges:
        if isinstance(e, tuple):
            edge_map[e] = _scrub("|".join(map(str, e)))
        else:
            edge_map[e] = _scrub(e)

    # Node label map
    node_map = {n: _scrub(n) for n in G.nodes}

    # Incidence via robust collector
    raw_inc = _collect_incidence(G)
    incidence = {edge_map[e]: [node_map[n] for n in raw_inc[e]] for e in raw_inc}

    # Properties
    try:
        props_df = _flatten_props_df(getattr(G._E, "properties", None))
    except Exception:
        props_df = pd.DataFrame(index=pd.Index([], name="edge"))

    if len(props_df.index) > 0:
        props_df.index = pd.Index([_scrub(ix) for ix in props_df.index], name="edge")

    # Align props to incidence keys exactly
    edges = pd.Index(list(incidence.keys()), name="edge")
    props_df = props_df.loc[props_df.index.intersection(edges)]
    missing = edges.difference(props_df.index)
    if len(missing):
        props_df = pd.concat([props_df, pd.DataFrame(index=missing)], axis=0)
    props_df = props_df.reindex(edges)

    return hnx.Hypergraph(incidence, edge_properties=props_df)

def safe_union(H: hnx.Hypergraph, G2: hnx.Hypergraph) -> hnx.Hypergraph:
    Hn  = normalize_hnx_graph_rebuild(H)
    G2n = normalize_hnx_graph_rebuild(G2)
    return Hn.union(G2n)

#END SANITIZE


def add_new_hypersubgraph_from_text(
    txt=None, generate=None, generate_figure=None,
    image_list=None, node_embeddings=None, tokenizer=None,
    model=None, original_graph=None, data_dir_output='./data_temp/',
    graph_root='graph_root', chunk_size=10000, chunk_overlap=2000,
    do_update_node_embeddings=True, do_distill=True, do_relabel=False, do_simplify_graph=True,
    size_threshold=10, repeat_refine=0, similarity_threshold=0.95,
    do_Louvain_on_new_graph=True, return_only_giant_component=False,
    save_common_graph=False, G_to_add=None, graph_pkl_to_add=None, sub_dfs=None,
    verbatim=True,
):
    display(Markdown(txt[:32] + "..."))

    updated_sub_dfs = sub_dfs #if no update is made 

    start_time = time.time()
    if verbatim:
        print("Now create or load new hypergraph...")

    # Determine hypergraph to add
    if G_to_add is not None and graph_pkl_to_add is not None:
        print("Provide only one of G_to_add or graph_pkl_to_add.")
        return
    elif graph_pkl_to_add is None and G_to_add is None:
        # Generate new hypergraph
        print("Make new hypergraph from text...")
        graph_pkl_to_add, G_to_add, _, _ = make_hypergraph_from_text(
            txt, generate, data_dir=data_dir_output,
            graph_root=graph_root, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, repeat_refine=repeat_refine,
            verbatim=verbatim,
        )
        if verbatim:
            print(f"Received new PKL from make_hypergraph_from_text: {graph_pkl_to_add}")
    elif G_to_add is None:
        # Load existing PKL
        if verbatim:
            print(f"Loading hypergraph from PKL; txt ignored: {graph_pkl_to_add}")
        with open(graph_pkl_to_add, 'rb') as f:
            G_to_add = pickle.load(f)

    print(f"--- Load/generate time: {time.time() - start_time:.2f}s ---")
    print("Now grow the existing hypergraph...")

    # Load original hypergraph (PKL or object)
    if isinstance(original_graph, str):
        with open(original_graph, 'rb') as f:
            H = pickle.load(f)
    else:
        H = deepcopy(original_graph)

    # Merge hypergraphs via HyperNetX union
    G_new = H.union(G_to_add)
    #G_new = safe_union(H, G_to_add)

    if do_update_node_embeddings:
        if verbatim:
            print("Updating node embeddings...")
        node_embeddings = update_hypernode_embeddings(
            node_embeddings, G_new, tokenizer, model, verbatim=verbatim
        )
        

    # 2) After simplification (if it itself updates embeddings)
    if do_simplify_graph:
        if verbatim:
            print("Simplifying hypergraph...")
        G_new, node_embeddings, updated_sub_dfs = simplify_hypergraph(
            G_new, sub_dfs, node_embeddings, tokenizer, model,
            similarity_threshold=similarity_threshold,
            use_llm=False, data_dir_output=data_dir_output,
            verbatim=verbatim,
        )
        

    # 3) After pruning small fragments
    if size_threshold > 0:
        if verbatim:
            print("Removing small fragments...")
        G_new, updated_sub_dfs = remove_small_hyperfragments(
            H_new=G_new,
            sub_dfs=updated_sub_dfs,
            size_threshold=size_threshold,
            return_singletons=False
        )
        node_embeddings = update_hypernode_embeddings(
            node_embeddings, G_new, tokenizer, model, verbatim=verbatim
        )


    # Giant component only
    if return_only_giant_component:
        if verbatim:
            print("Selecting giant component...")
        cc = sorted(hnx.connected_components(G_new), key=len, reverse=True)
        G_new = G_new.subhypergraph(cc[0])
        node_embeddings = update_hypernode_embeddings(
            node_embeddings, G_new, tokenizer, model, verbatim=verbatim
        )

    # Louvain clustering
    if do_Louvain_on_new_graph:
        G_new = graph_Louvain(G_new)
        if verbatim:
            print("Completed Louvain clustering.")

    if verbatim:
        print("Hypergraph update complete.")

    # Save integrated hypergraph as PKL
    integrated_pkl = f"{data_dir_output}/{graph_root}_integrated.pkl"
    if verbatim:
        print(f"Saving integrated hypergraph to: {integrated_pkl}")
    with open(integrated_pkl, 'wb') as f:
        pickle.dump(G_new, f)
    if verbatim:
        print("Integrated hypergraph saved.")

    #save the updated sub_dfs 
    updated_subdfs_pkl = f"{data_dir_output}/updated_sub_dfs.pkl"
    if verbatim:
        print(f"Saving updated subdfs to: {updated_subdfs_pkl}")
    with open(updated_subdfs_pkl, 'wb') as f:
        pickle.dump(updated_sub_dfs, f)
    if verbatim:
        print("Updated subdfs are saved.")

    print(G_new, integrated_pkl)
    return integrated_pkl, G_new, G_to_add, node_embeddings, updated_sub_dfs

