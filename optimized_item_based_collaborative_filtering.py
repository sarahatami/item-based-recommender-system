# Sara Hatami
# Recommender Systems
# An optimized item‑based collaborative filtering algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

pd.set_option('display.width', 200000)
pd.set_option('max_columns', 200000)
pd.set_option('display.max_rows', None)


def optimized_item_based_cf():
    # get data
    ratings_df = pd.read_csv(r'E:\MASTER\Uni\Term1\RS\project\ml-100k\u.data', sep='\t') #99999*4
    ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # create rating matrix
    rating_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating')
    users_mu = rating_matrix.mean(axis=1)
    rating_mean_subtracted = rating_matrix - users_mu[:, None]

    def adjusted_cosine_sim(i1, i2, a):
        if i1 != i2:
            # n & m
            notnull_ratings = rating_mean_subtracted[[i1, i2]][(rating_mean_subtracted[i1].notna()) &
                                                               (rating_mean_subtracted[i2].notna())]
            n = notnull_ratings.shape[0]

            onenull_ratings = rating_mean_subtracted[[i1, i2]][(rating_mean_subtracted[i1].notna()) |
                                                               (rating_mean_subtracted[i2].notna())]
            m = onenull_ratings.shape[0]
            # calculate balancing factor
            balancing_factor = 1 - a*(1 - n/m)
            # calculate adjusted cosine similarity
            cos_sim = balancing_factor * (np.dot(notnull_ratings[i1], notnull_ratings[i2])
                                          / (np.linalg.norm(notnull_ratings[i1]) * np.linalg.norm(notnull_ratings[i2])))
            return cos_sim


    def get_k_neighbors(base_item_num, k , a):
        similarities = []
        for item_name, item in rating_mean_subtracted.iteritems():
            sim = adjusted_cosine_sim(base_item_num, item_name, a)
            if sim != None:
                similarities.append((item_name, sim))
        # sort similarities & return top k
        top_k_tupple = sorted(similarities, key=lambda t: t[1], reverse=True)[:k]
        return top_k_tupple


    def estimate_rating(u, i, k, a):
        top_k_items = [x[0] for x in get_k_neighbors(i, k, a)]
        top_k_sims = [x[1] for x in get_k_neighbors(i, k, a)]

        rate_u_to_top_k = rating_matrix.loc[u][top_k_items]
        df = pd.DataFrame({'rates': rate_u_to_top_k, 'sims': top_k_sims})
        notnull_df = df[['rates', 'sims']][(df['rates'].notna()) & (df['sims'].notna())]
        rate_u_to_i = np.dot(notnull_df['rates'], notnull_df['sims']) / sum(abs(notnull_df['sims']))
        # print('rate_u_to_i: ', u, ' ', i, ' ', a, ' ', rate_u_to_i)
        return rate_u_to_i


    def create_estimated_ratings_matrix(k , a):
        x, y = rating_matrix.shape[0], rating_matrix.shape[1]
        estimated_matrix = np.zeros((x, y))
        for i in range(1, x+1):
            for j in range(1, y+1):
                if not (pd.isnull(rating_matrix.loc[i][j])):
                    estimated_matrix[i-1][j-1]=estimate_rating(i, j, k, a)
        estimated_matrix[np.isnan(estimated_matrix)] = 0
        return estimated_matrix


    def calculate_error(k, a):
        # number of ratings
        n = (rating_matrix.shape[0]*rating_matrix.shape[1]) - rating_matrix.isna().sum().sum()
        rating_matrix_np = rating_matrix.fillna(0).to_numpy()
        mae = (np.sum(abs(rating_matrix_np-create_estimated_ratings_matrix(k, a))) / n) /4
        # print('k:', k, ' ', 'a:', a)
        return mae


    # create final dataframe
    error_df = pd.DataFrame(columns=['k', 'mae_0', 'mae_0.2', 'mae_0.4', 'mae_0.6', 'mae_0.8', 'mae_1'])
    for i in (4, 8, 12, 16, 20):
        error_df.loc[i] = [i] + [calculate_error(i, 0)] + [calculate_error(i, 0.2)] + [calculate_error(i, 0.4)] \
                          + [calculate_error(i, 0.6)] + [calculate_error(i, 0.8)] + [calculate_error(i, 1)]

    # plot final result
    plt.plot(error_df['k'], error_df['mae_0'], marker=".", color='blue', label='a=0')
    plt.plot(error_df['k'], error_df['mae_0.2'], marker=".", color='pink', label='a=0.2')
    plt.plot(error_df['k'], error_df['mae_0.4'], marker=".", color='yellow', label='a=0.4')
    plt.plot(error_df['k'], error_df['mae_0.6'], marker=".", color='cyan', label='a=0.6')
    plt.plot(error_df['k'], error_df['mae_0.8'], marker=".", color='purple', label='a=0.8')
    plt.plot(error_df['k'], error_df['mae_1'], marker=".", color='red', label='a=1')
    plt.xlabel('No. of Neighbors')
    plt.ylabel('MAE')
    plt.legend(loc='best')
    plt.show()



if __name__ == '__main__':
    t1=datetime.datetime.now()
    optimized_item_based_cf()
    t2=datetime.datetime.now()
    dist=t2-t1
    print('started at :',t1)
    print('finished at:',t2,' | elapsed time (s):',dist.seconds)










