import numpy as np
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
import gensim.downloader
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import torch
from transformers import AutoTokenizer, AutoModel
from langdetect import detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# nltk.download("stopwords")
# nltk.download("punkt_tab")
# nltk.download("vader_lexicon")

class Features:
    def __init__(self, embeddings: str, **kwargs):
        '''
        TODO:
        0. Merge: SELECT * FROM train_stances LEFT JOIN train_bodies ON train_stances.body_id = train_bodies.body_id
        1. Preperation
            1.1 Normalize the texts (to lowercase, remove extra spaces and punctuation, tokenize)
            1.2 prepare the corpus based on the whole training set, i.e. pairs of headline and body. 
        2. Generate word embeddings, using word2vec in our case.
            2.1 not necessary to remove stop words (the embeddings for stop words can capture contextual relationships that might still be relevant), but I will do it anyways. IDGAF.
            2.2 Pretrained model (Google news, gensim) or train our model using the dataset?
            2.3 The embedding vector of a n-gram is the avg of every word. i.e. {"covid-19 vaccine": [.1, .2, .4], "vaccine approved": [-.5, 1.4, -.6]}
        3. TF-IDF feature engineering
            3.1 use the corpus to fit a tf-idf vectorizor. it should assign a tf-idf score to each n-gram, i.e. {"covid-19 vaccine": .4, "vaccine approved": .6}
                ** We may feed the raw passages to the vectorizer, no need to normalize beforehand. **
                ** Question: what if new n-grams appear in the test set? by default, the score is set to zero. but important info could be ignored by doing so. **
                ** can remove stop words and punctuations when initializing the vectorizer **
            3.2 the TF-IDF vector of each sample has a length of #(unique n-grams) in the corpus. usually sparse, with high dimension, may cause vanishing gradient. Solutions needed.
                ** Consider SVD or explicitly set max_features **
        4. Joint only features we can consider:
            4.1 Jaccard similarity (n-grams overlap): suppose A, B are the set of n-grams in the body & the headline. J(A, B) = |A and B| / |A or B| (# of common n-grams over # of unique n-grams accross both texts)
            4.2 cosine similarity of word2vec embeddings: Similarity = (headline_vec * body_vec) / [norm(headline_vec) * norm(body_vec)]
            4.3 cosine similarity of tf-idf vectors
            4.4 cosine similarity of tf-idf **score** weighted embedding vectors:
                4.4.1 weight word2vec embeddings for each n-gram in step 2 by its tf-idf score. just multiply them.
                4.4.2 the embedding of the headline/body is the average of the weighted embeddings for all n-grams in the headline/body **those can be individual features**
                4.4.3 compute similarity score using cosine similarity
        5. Other Individual/joint features we can consider:
            5.1 average word2vec embedding
            5.2 count features: sklearn
            5.3 word length features: min, max, avg
            5.4 lexical diversity: unique words / total words
            5.5 sentiment: pretrained models, VADER, BERT (transformer-based), or train our own sentiment model
                ** Keep all the stop words and punctuations in the aforementioned pretrained models, because the embeddings for those words still carry useful semantic or syntactic information **
        6. Dimension Reduction
            6.1 Methods to be considered: SVD, PCA
        '''
        self.n_gram = kwargs.get("n_gram", 1)
        self.stopword = kwargs.get("stopword", False)
        self.wordvec = gensim.downloader.load(embeddings)

    def train(self):
        self.training = True
    
    def test(self):
        self.training = False

    def merge(self, body_path, stance_path):
        '''
        TODO 0
        '''
        body = pd.read_csv(body_path)
        stance = pd.read_csv(stance_path)
        body["articleBody_token"] = body["articleBody"].apply(self.normalize_text, args = (self.stopword, ))
        stance["Headline_token"] = stance["Headline"].apply(self.normalize_text, args = (self.stopword, ))

        body = body.loc[body["articleBody"].apply(lambda x: detect(x) == "en"), :]

        data = pd.merge(left=body, right=stance, how = "left", on = "Body ID").drop(columns = "Body ID")

        assert data.isna().sum().sum() == 0
        self.data = data
    
    @staticmethod
    def normalize_text(text, rm_stopword):
        '''
        TODO 1.1
        '''
        # text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # remove punctuation
        tokens = nltk.tokenize.word_tokenize(text)
        if rm_stopword:
            stopwords = set(nltk.corpus.stopwords.words('english'))
        else:
            stopwords = set()
        tokens = [word for word in tokens if word not in stopwords]
        return tokens
    
    @staticmethod
    def get_embedding(tokens, model):
        '''
        TODO 2.1
        '''
        embeddings = np.zeros((len(tokens), model.vector_size))
        for i, token in enumerate(tokens):
            try:
                embeddings[i, :] = model[token]
            except KeyError:
                pass
        return np.mean(embeddings, axis = 0)

    def word2vec(self):
        '''
        TODO 2.2 & 5.1
        '''
        return np.vstack(self.data["articleBody_token"].apply(self.get_embedding, args = (self.wordvec,)).tolist()), \
                np.vstack(self.data["Headline_token"].apply(self.get_embedding, args = (self.wordvec,)).tolist()), \
                np.vstack((self.data["articleBody_token"] + self.data["Headline_token"]).apply(self.get_embedding, args = (self.wordvec,)).tolist())
    
    def tf_idf(self):
        '''
        TODO 3.x
        '''
        if self.training:
            tfidf_vectorizer = TfidfVectorizer(preprocessor=lambda text: re.sub(r'[^\w\s]', '', text),
                                                stop_words = "english", # always remove stopwords
                                                max_features = 300)
            tfidf_vectorizer.fit(self.data["articleBody"] + " " + self.data["Headline"])
            self.tfidf_vectorizer = tfidf_vectorizer
        assert hasattr(self, "tfidf_vectorizer")
        return self.tfidf_vectorizer.transform(self.data["articleBody"]).toarray(), self.tfidf_vectorizer.transform(self.data["Headline"]).toarray(),\
        self.tfidf_vectorizer.transform(self.data["articleBody"] + " " + self.data["Headline"]).toarray()
    
    def count_feature(self):
        '''
        TODO 5.2
        '''
        if self.training:
            count_vectorizer = CountVectorizer(preprocessor=lambda text: re.sub(r'[^\w\s]', '', text),
                                                stop_words = "english", # always remove stopwords
                                                max_features = 300)
            count_vectorizer.fit(self.data["articleBody"] + " " + self.data["Headline"])
            self.count_vectorizer = count_vectorizer
        assert hasattr(self, "tfidf_vectorizer")
        return self.count_vectorizer.transform(self.data["articleBody"]).toarray(), self.count_vectorizer.transform(self.data["Headline"]).toarray(),\
                self.count_vectorizer.transform(self.data["articleBody"] + " " + self.data["Headline"]).toarray()

    def jaccard_sim(self):
        '''
        TODO 4.1
        '''
        def compute_union_intersect(row):
            return len(set(row['Headline_token']) & set(row['articleBody_token'])) / len(set(row['Headline_token']) | set(row['articleBody_token']))
        return self.data.apply(compute_union_intersect, axis = 1).to_numpy().reshape(-1, 1)
    
    @staticmethod
    def cosine_sim(matA, matB):
        '''
        TODO 4.2
            ** Problem: embeddings or tf-idf vecs may have zero (or close to zero) norms. How to fix this? **
        '''
        assert matA.shape == matB.shape
        return np.nan_to_num((np.sum(matA * matB, axis = 1) / (np.linalg.norm(matA, axis = 1) * np.linalg.norm(matB, axis = 1))), copy = False, nan = 0.0).reshape(-1, 1)

    @staticmethod
    def word_length_diversity(text):
        '''
        TODO 5.3 & 5.4
        '''
        def get_length(row):
            arr = np.fromiter(map(len, row), dtype = int)
            return [arr.max(), arr.min(), arr.mean(), len(set(row))/len(row)]
        return np.vstack(text.apply(get_length).tolist())

    @staticmethod
    def vader_score(text):
        '''
        TODO 5.5
        '''
        sid = SentimentIntensityAnalyzer()
        def get_vader_score(row):
            scores = sid.polarity_scores(row) # neg, neu, pos, compound
            return list(scores.values())
        return np.vstack(text.apply(get_vader_score).tolist())
    
    @staticmethod
    def get_optimal_dim(decomp_model):
        cum_var = np.cumsum(decomp_model.explained_variance_ratio_)
        return np.argmax(np.abs((cum_var[-1] - cum_var[0]) * np.arange(1, cum_var.shape[0] + 1) - (cum_var.shape[0] - 1) * cum_var + cum_var.shape[0]*cum_var[0] - cum_var[-1])\
                / np.sqrt((cum_var[-1] - cum_var[0])**2 + (cum_var.shape[0] - 1)**2)) + 1

    @staticmethod
    def rescale(body_data, head_data, combined_data):
        body_scaler, head_scaler, combined_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

        body_scaler.fit(body_data)
        head_scaler.fit(head_data)
        combined_scaler.fit(combined_data)

        return body_scaler, head_scaler, combined_scaler

    @staticmethod
    def fit_decomp(data, method, n_comp):
        if method == "svd":
            decomp_model = TruncatedSVD(n_comp)
        elif method == "pca":
            decomp_model = PCA(n_comp)
        else:
            raise NotImplementedError("Unsupported method '%s'. Choose 'svd' or 'pca'." %(method))
        decomp_model.fit(data)
        return decomp_model

    @classmethod
    def dim_reduction(cls, body_data, head_data, combined_data, method: str, n_comp: int|str, feature_name: str, verbose = 1):
        n_features = body_data.shape[1]
        
        # if method == "svd":
        #     body_decomp, head_decomp, combined_decomp = TruncatedSVD(n_components), TruncatedSVD(n_components), TruncatedSVD(n_components)
        # elif method == "pca":
        #     body_decomp, head_decomp, combined_decomp = PCA(n_components), PCA(n_components), PCA(n_components)

        # body_decomp.fit(body_data)
        # head_decomp.fit(head_data)
        # combined_decomp.fit(combined_data)

        body_decomp = cls.fit_decomp(body_data, method, n_features)
        head_decomp = cls.fit_decomp(head_data, method, n_features)
        combined_decomp = cls.fit_decomp(combined_data, method, n_features)

        if n_comp == "auto":
            n_comp_opt = [cls.get_optimal_dim(body_decomp), cls.get_optimal_dim(head_decomp), cls.get_optimal_dim(combined_decomp)]
        else: 
            n_comp_opt = [n_comp, n_comp, n_comp]

        if verbose:
            print("Selected dims for {} feature: {}".format(feature_name, n_comp_opt))
        
        return body_decomp, head_decomp, combined_decomp, n_comp_opt
    
    def reduce_count(self, body_data, head_data, combined_data):
        if self.training:
            self.body_scaler_count, self.head_scaler_count, self.combined_scaler_count = self.rescale(body_data, head_data, combined_data)
        
        body_data = self.body_scaler_count.transform(body_data)
        head_data = self.head_scaler_count.transform(head_data)
        combined_data = self.combined_scaler_count.transform(combined_data)
        
        if self.training:
            self.body_decomp_count, self.head_decomp_count, self.combined_decomp_count, self.n_comp_count = self.dim_reduction(body_data, head_data, combined_data, 
                                                                                                                               method = "svd", 
                                                                                                                               n_comp = "auto", 
                                                                                                                               feature_name = "Count")
        
        body_data = self.body_decomp_count.transform(body_data)[:, :self.n_comp_count[0]]
        head_data = self.head_decomp_count.transform(head_data)[:, :self.n_comp_count[1]]
        combined_data = self.combined_decomp_count.transform(combined_data)[:, :self.n_comp_count[2]]

        return body_data, head_data, combined_data

    def reduce_tfidf(self, body_data, head_data, combined_data):
        if self.training:
            self.body_scaler_tfidf, self.head_scaler_tfidf, self.combined_scaler_tfidf = self.rescale(body_data, head_data, combined_data)
        
        body_data = self.body_scaler_tfidf.transform(body_data)
        head_data = self.head_scaler_tfidf.transform(head_data)
        combined_data = self.combined_scaler_tfidf.transform(combined_data)
        
        if self.training:
            self.body_decomp_tfidf, self.head_decomp_tfidf, self.combined_decomp_tfidf, self.n_comp_tfidf = self.dim_reduction(body_data, head_data, combined_data, 
                                                                                                                               method = "svd", 
                                                                                                                               n_comp = 150,
                                                                                                                               feature_name = "TF-IDF")
        
        body_data = self.body_decomp_tfidf.transform(body_data)[:, :self.n_comp_tfidf[0]]
        head_data = self.head_decomp_tfidf.transform(head_data)[:, :self.n_comp_tfidf[1]]
        combined_data = self.combined_decomp_tfidf.transform(combined_data)[:, :self.n_comp_tfidf[2]]

        return body_data, head_data, combined_data

    def bert_embedding(self, data):
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        def get_bert(text):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            return outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()
        return np.vstack(data.apply(get_bert).tolist())



    def write(self, file_name, **arrays: np.array):
        # chunk_size = 100
        # chunk_idx = [(i * chunk_size, min((i + 1) * chunk_size, n_rows)) for i in range((n_rows + chunk_size - 1) // chunk_size)]
        # with open("./features/feature_{}.csv".format(file_name), 'w') as f:
        #     for start, end in chunk_idx:
        #         chunk = np.hstack([arr[start:end, :] for arr in arrays.values()])
        #         np.savetxt(f, chunk, delimiter = ',', mode = 'a', fmt = "%f")

        np.save("./features/feature_aug_{}.npy".format(file_name), np.hstack(list(arrays.values())))
        
        if self.training:
            readme_content = []
            curr_col = 0
            for name, arr in arrays.items():
                num_cols = arr.shape[1]
                readme_content.append("Column %d - %d: %s" %(curr_col, curr_col + num_cols - 1, name))
                curr_col += num_cols

            with open("./features/README.txt", 'w') as f:
                f.write("\n".join(readme_content))
        
        self.data["Stance"].to_csv("./features/label_{}.csv".format(file_name))

    def fit(self, body_path, stance_path):
        '''
        print the cols corresponding to each feature to readme
        '''
        self.merge(body_path, stance_path)
        jaccard = self.jaccard_sim()

        body_bert = self.bert_embedding(self.data["Headline"])

        body_embed, head_embed, combined_embed = self.word2vec()
        embed_sim = self.cosine_sim(body_embed, head_embed)

        # dimension reduction needed for the following lines
        body_tfidf, head_tfidf, combined_tfidf = self.reduce_tfidf(*self.tf_idf())

        tfidf_sim = self.cosine_sim(body_tfidf, head_tfidf)

        body_count, head_count, combined_count = self.reduce_count(*self.count_feature()) # oh boy those variable names look creepy. body_count? hell no.

        body_len_div = self.word_length_diversity(self.data["articleBody_token"])
        head_len_div = self.word_length_diversity(self.data["Headline_token"])
        combined_len_div = self.word_length_diversity(self.data["articleBody_token"] + self.data["Headline_token"])

        body_sentiment = self.vader_score(self.data["articleBody"])
        head_sentiment = self.vader_score(self.data["Headline"])
        combined_sentiment = self.vader_score(self.data["articleBody"] + " " + self.data["Headline"])

        body_bert = self.bert_embedding(self.data["articleBody"])
        head_bert = self.bert_embedding(self.data["Headline"])
        combined_bert = self.bert_embedding(self.data["articleBody"] + " " + self.data["Headline"])
        bert_sim = self.cosine_sim(body_bert, head_bert)

        self.write(file_name = "train" if self.training else "test",
                   body_embedding = body_embed, body_tfidf = body_tfidf, body_wordcount = body_count, body_length_diversity = body_len_div, body_sentiment = body_sentiment, body_bert = body_bert,
                   head_embedding = head_embed, head_tfidf = head_tfidf, head_wordcount = head_count, head_length_diversity = head_len_div, head_sentiment = head_sentiment, head_bert = head_bert,
                   combined_embedding = combined_embed, combined_tfidf = combined_tfidf, combined_wordcount = combined_count, combined_length_diversity = combined_len_div, combined_sentiment = combined_sentiment, 
                   combined_bert = combined_bert, embed_similarity = embed_sim, tfidf_similarity = tfidf_sim, jaccard_similarity = jaccard, bert_similarity = bert_sim)


def main():
    feature_set = Features(embeddings = "word2vec-google-news-300", stopword = True)
    feature_set.train()
    feature_set.fit(body_path = "./fnc-1-master/train_bodies.csv", stance_path = "./fnc-1-master/train_stances.csv")
    feature_set.test()
    feature_set.fit(body_path = "./fnc-1-master/competition_test_bodies.csv", stance_path = "./fnc-1-master/competition_test_stances.csv")

if __name__ == "__main__":
    main()