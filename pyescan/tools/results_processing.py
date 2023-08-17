# Grouby patient number
# and then group by another value and accregate predictions
def get_results_from_dataframe(df, classes=None, group_on='file.path', patient_id_key='patient.number'):
    import numpy as np
    
    if classes is None:
        classes = [ cls.replace("pred.","") for cls in df.columns
                    if cls.startswith("pred.")
                    and not cls in ["pred.class", "pred.model"] ]

    pandas_query = [ 'pred.'+cls for cls in classes ]
    
    truth, pred_scores, pat_ids = list(), list(), list()
    for pat_id, pat_entries in df.groupby(patient_id_key):
        gene = pat_entries['gene'].values[0]
        gene_ind = classes.index(gene)
        for val, grouped_entries in pat_entries.groupby(group_on):
            pred_scores.append(grouped_entries[pandas_query].mean(axis=0))
            pat_ids.append(pat_id)
            truth.append(gene_ind)   
    return np.array(truth), np.array(pred_scores), pat_ids, classes