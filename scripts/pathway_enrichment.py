import numpy as np
import pandas as pd
import gseapy as gp
from tqdm import tqdm
import warnings
import sys
sys.path.append('./TransPath')
sys.path.append('./src')
from typing import List, Optional
from scipy.stats import rankdata, norm

warnings.filterwarnings('ignore')


def extract_pathway_genes(library: str = 'KEGG_2021_Human') -> dict:
    """Extract pathway genes from GSEApy KEGG library"""
    print(f"Extracting pathway genes from {library}...")
    
    # Get all pathways from the library
    all_pathways = gp.get_library(library)
    return all_pathways

def wilcoxon_vectorized(X,ranks, gene_mask):
    """
    Vectorized Wilcoxon rank-sum (Mann–Whitney U test) per column.

    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix of shape (n_genes, n_cells).
    gene_mask : array-like of bool
        Boolean mask of length n_genes indicating genes in gene set.

    Returns
    -------
    U : np.ndarray
        Array of shape (n_cells,) containing the U statistic for each cell.
    pvals : np.ndarray
        Two-sided p-value for each cell.
    """

    gene_mask = np.asarray(gene_mask, dtype=bool)

    n_genes, n_cells = X.shape
    n1 = gene_mask.sum()
    n2 = n_genes - n1

    R1 = np.sum(ranks[gene_mask, :], axis=0)
    U = R1 - n1 * (n1 + 1) / 2.0

    mean_U = n1 * n2 / 2.0
    std_U = np.sqrt(n1 * n2 * (n_genes + 1) / 12.0)

    z = (U - mean_U) / std_U
    pvals = norm.sf(z)  

    return U, pvals

def calculate_pathway_enrichment(expression_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pathway enrichment scores using expression DataFrame
    
    Args:
        expression_df: DataFrame with cells as rows and genes as columns
        
    Returns:
        DataFrame with enrichment scores (cells x pathways)
    """
    print(f"Expression data shape: {expression_df.shape}")
    
    # Extract pathway genes
    pathway_genes = extract_pathway_genes()
    
    # Filter genes that exist in data
    filtered_pathways = {}
    for pathway_name, genes in pathway_genes.items():
        genes_in_data = [gene for gene in genes if gene in expression_df.columns]
        if len(genes_in_data) >= 5:
            filtered_pathways[pathway_name] = genes_in_data
            print(f"{pathway_name}: {len(genes_in_data)} genes found")
    
    if not filtered_pathways:
        raise ValueError("No pathway genes found in data!")
    
    print(f"Using {len(filtered_pathways)} pathways for enrichment")
    
    # Calculate enrichment scores for each pathway
    enrichment_results = {}
    enrichment_U = {}
    #rank_df = expression_df.T.rank(ascending=False).T
    X = expression_df.T.values
    ranks = rankdata(X, axis=0, method='average')
    for pathway_name, pathway_genes in tqdm(filtered_pathways.items()):
        U, enrichment_score = wilcoxon_vectorized(X, ranks, expression_df.T.index.isin(pathway_genes)) #rank_df[pathway_genes].mean(axis=1) 
        enrichment_results[pathway_name] = enrichment_score
        enrichment_U[pathway_name] = U
    enrichment_df = pd.DataFrame(enrichment_results, index=expression_df.index)
    enrichment_U_df = pd.DataFrame(enrichment_U, index=expression_df.index)
    ranks_df = pd.DataFrame(ranks.T, index=expression_df.index, columns=expression_df.T.index)

    print(f"Enrichment analysis completed! Shape: {enrichment_df.shape}")
    return enrichment_df, enrichment_U_df

def analyze_scanpy_object(adata) -> pd.DataFrame:
    """
    Analyze Scanpy AnnData object and return enrichment DataFrame
    
    Args:
        adata: Scanpy AnnData object
        
    Returns:
        DataFrame with enrichment scores
    """
    # Convert to DataFrame
    expression_df = adata.to_df()
    
    # Calculate enrichment
    enrichment_df, rank_df = calculate_pathway_enrichment(expression_df)
    
    return enrichment_df, rank_df