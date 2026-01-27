from typing import List, Dict, Any

def reciprocal_rank_fusion(results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Standard Reciprocal Rank Fusion (RRF) algorithm.
    
    Args:
        results_list: A list of result lists, where each inner list contains dictionaries 
                      representing a document (must have an 'id' or unique 'content').
        k: A constant to prevent small ranks from dominating.
        
    Returns:
        A sorted list of merged results with 'rrf_score'.
    """
    fused_scores = {}
    
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            # Use 'content' or a unique ID as key
            doc_id = result.get('id') or result.get('content')
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0.0, "data": result}
            
            fused_scores[doc_id]["score"] += 1.0 / (k + rank)
            
    # Sort by score descending
    sorted_results = sorted(
        fused_scores.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    # Return flattened list with the new score
    final_results = []
    for item in sorted_results:
        doc = item["data"].copy()
        doc["rrf_score"] = item["score"]
        final_results.append(doc)
        
    return final_results
