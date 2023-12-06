import pandas as pd
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score


def read_data(data_file):
    return pd.read_csv(data_file)


def ndcg_score_custom(df_with_qid, scores, k):
    df_with_score = df_with_qid.copy()
    df_with_score['score'] = scores
    dictionary = {}
    summation = 0
    count = 0
    for name, group in df_with_score.groupby(['qid']):
        qid = group['qid'].min()
        df_qid = df_with_score[df_with_score['qid'] == qid]

        rel = [df_qid['C'].tolist()]
        rank = [df_qid['score'].tolist()]
        if len(rel) > 0 and len(rel[0]) > 1:
            res = ndcg_score(rel, rank, k=k)
            dictionary[qid] = res
            summation += res
            count += 1

    print("NDCG@" + str(k) + ": " + str(summation / count))

    return summation / count


def mrr_score_custom(df_with_qid, scores):
    df_with_score = df_with_qid.copy()
    df_with_score['score'] = scores
    summation = 0
    count = 0
    for name, group in df_with_score.groupby(['qid']):
        qid = group['qid'].min()
        df_qid = df_with_score[df_with_score['qid'] == qid]

        rel = df_qid['C'].tolist()
        rank = df_qid['score'].tolist()
        if len(rel) > 0:
            max_score = max(rank)
            max_index = rank.index(max_score)
            if max_index > -1:
                summation += (1 / (max_index + 1))
                count += 1

    print("MRR: " + str(summation / count))

    return summation / count
