import pandas as pd
from tqdm.autonotebook import tqdm


class GenericRecommender:

    def __init__(self, dataset_metadata, model, top_n: int = 10):
        self.top_n = top_n
        self.dataset = dataset_metadata.dataset
        self.model = model
        self.catalogue = set(self.dataset['itemId'])

    def recommend_all(self, full_sort = False):
        """
        Get all recommendations.
        :param top_n:
        :return: recommendations for any user.
        """

        sorted_df = self.dataset.sort_values(by=["userId"])
        ratings = sorted_df.groupby('userId', sort=True)

        if full_sort:
            item_model_df = self.model.recoxplainer_dataset.original_item_id.copy()
            item_model_df["model_id"] = item_model_df["item_id"].apply(str)\
                .map(self.model.recbole_dataset.field2token_id[self.model.recbole_dataset.iid_field])
            model_item_map_df = item_model_df.reset_index().set_index("model_id")

            # Process users in batches to manage memory usage
            uids = sorted_df["userId"].unique().tolist()
            num_users = len(uids)
            batch_size = 1000  # Adjust based on available memory
            recommendations_df = pd.DataFrame({'userId': [], 'itemId': [], 'rank': []})
            
            for batch_start in tqdm(range(0, num_users, batch_size), desc="Processing batches"):
                batch_end = min(batch_start + batch_size, num_users)
                batch_uids = uids[batch_start:batch_end]

                scores = self.model.full_sort_predict(batch_uids)

                i = 0
                recommendations_data = []
                for user_id, user_ratings in ratings:
                    unrated = self.get_unrated(user_ratings['itemId'])
                    item_model_df.loc[unrated]
                    recommendations = scores[i][item_model_df.loc[unrated]["model_id"].tolist()].topk(10)
                    recommendations = model_item_map_df['itemId'].iloc[recommendations[1].tolist()].tolist()
                    recommendations_data += [
                        {
                            "userId": user_id,
                            "itemId": item_id,
                            "rank": rank+1
                        }
                        for rank, item_id in enumerate(recommendations)
                    ]
                    i += 1
                recommendations_df = pd.concat([
                    recommendations_df, pd.DataFrame(recommendations_data).astype(float)],
                    ignore_index=True)
            return recommendations_df

        recommendations = pd.DataFrame({'userId': [], 'itemId': [], 'rank': []})

        with tqdm(total=self.dataset['userId'].nunique(), desc="Recommending for users: ") as pbar:
            for user_id, user_ratings in ratings:
                recommendations = pd.concat([
                    recommendations,
                    self.recommend_user(user_id, user_ratings)],
                    ignore_index=True)
                pbar.update()

        return recommendations

    def rank_prediction(self, user_id, target_item_id, predictions):
        recommendations = pd.DataFrame({'userId': user_id,
                                        'itemId': target_item_id,
                                        'prediction': predictions})

        recommendations['rank'] = recommendations['prediction'] \
            .rank(method='first', ascending=False)

        recommendations \
            .sort_values(['userId', 'rank'], inplace=True)

        recommendations = recommendations[recommendations['rank'] <= self.top_n]

        return recommendations[['userId', 'itemId', 'rank']]

    def get_unrated(self, user_ratings):
        """
        Extract the set of items a user has not rated.
        :param user_ratings: list, items rated.
        :return: list, items not rated.
        """
        unrated_item_id = self.catalogue - set(user_ratings)
        unrated_item_id = list(unrated_item_id)
        return unrated_item_id

    def get_rated(self, user_id):
        """
        Extract the set of items a user has not rated.
        :param user_id: userId rated.
        :return: list, rated items.
        """
        rated = self.dataset[self.dataset['userId'] == user_id]
        return rated
