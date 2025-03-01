import re, nltk, indicnlp, pandas as pd, string, numpy as np, torch, fasttext, os
from bpemb import BPEmb
from langdetect import detect
from indicnlp.tokenize import indic_tokenize
import stopwordsiso as stopwords
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertTokenizer, BertModel

vectormodelpath = '../fastText/vectormodels'
fasttext_bn = fasttext.load_model(f'{vectormodelpath}/cc.bn.300.bin')
fasttext_hi = fasttext.load_model(f'{vectormodelpath}/cc.hi.300.bin')
fasttext_en = fasttext.load_model(f'{vectormodelpath}/cc.en.300.bin')

vocabsize = 10000
bpemb_en = BPEmb(lang = "en", dim = 300, vs=vocabsize)
bpemb_bn = BPEmb(lang = "bn", dim = 300, vs=vocabsize)
bpemb_hi = BPEmb(lang = "hi", dim = 300, vs=vocabsize)

labelmap = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
ftmodelmap = {'bn':fasttext_bn, 'hi':fasttext_hi, 'en':fasttext_en}
bpmodelmap = {'bn':bpemb_bn, 'hi':bpemb_hi, 'en':bpemb_en}

def returndataset(datasetnames):
    dataset_path = '../archive'
    data = []
    for name in datasetnames:
        filename = dataset_path + '/' + name + '.csv'
        try:
            dataframe = pd.read_csv(filename)
            dataframe = dataframe.dropna(axis=0, how='any')
            data.append(dataframe)
            print(f"loaded dataset: {name}")
        except FileNotFoundError:
            print(f"dataset not found: {name}")
        except Exception as e:
            print(f"error {e} occured while loading dataset: {name}")

    if len(data) != 0:
        data_set = pd.concat(data, ignore_index = True)
        # print(f"length of dataset {len(data_set)}")
        # print(data_set.head())
        return data_set
        #return data_set

def remove(text):
    text = re.sub(r'[^a-zA-Z\u0900-\u097F\u0980-\u09FF\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    tokens = indic_tokenize.trivial_tokenize(text)
    return tokens

def detectlanguage(word):
    try:
        detect(word)
    except:
        return 'en'

def fasttextembed(tokens):
    sentence_embedding = []
    for token in tokens:
        ln = detectlanguage(token)
        model = ftmodelmap.get(ln, fasttext_en)
        if model is not None:
            sentence_embedding.append(model.get_word_vector(token))
        else:
            sentence_embedding.append(np.zeros(300))
    sentence_embedding = np.mean(sentence_embedding, axis=0)
    # print(len(sentence_embedding))
    return sentence_embedding

def bpembembed(tokens):
    sentence_embedding = []
    for token in tokens:
        ln = detectlanguage(token)
        model = bpmodelmap.get(ln, bpemb_en)
        word_embeddings = model.embed(token)
        averaged_embeddings = word_embeddings.mean(axis=0)
        sentence_embedding.append(averaged_embeddings)
    sentence_embedding = np.mean(np.array(sentence_embedding), axis=0)
    return sentence_embedding

def bpembencode(tokens):
    idlist = []
    for token in tokens:
        ln = detectlanguage(token)
        model = bpmodelmap.get(ln, bpemb_en)
        ids = model.encode_ids(token)
        match ln:
            case 'en':
                idlist.extend(ids)
            case 'bn':
                shift = vocabsize
                idlist.extend([idn + shift for idn in ids])
            case _:
                shift = 2 * vocabsize
                idlist.extend([idn + shift for idn in ids])
    if len(idlist) < 300:
        idlist = idlist + [0] * (300 - len(idlist))
    else:
        if len(idlist) > 300:
            idlist = idlist[:300]
    return np.array(idlist)

def mBert(tokens):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    input_data = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
    output = model(**input_data)
    embeddings = output.last_hidden_state.mean(dim=1)
    embeddingsnparray = embeddings[0].detach().cpu().numpy()
    return embeddingsnparray

def process(data, embeddingmodel):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("using device ", device)
    y_array = []
    x_array = []
    allstopwords = stopwords.stopwords(["en", "bn", "hi"])
    for index, row in data.iterrows():
        print("iterating index", index)
        label = row['label']
        text = row['text']
        if embeddingmodel  == 'mbert':
                sentence_embedding = mBert(text)
                if sentence_embedding is not None:
                    x_array.append(sentence_embedding)
                    y_array.append(labelmap[label])
                else:
                    print("sentence embedding is None")
        elif embeddingmodel == 'fasttext':
            text = remove(text)
            tokens = tokenize(text)
            cleanedtokens = [word for word in tokens if word not in allstopwords and word not in string.punctuation] # removing punctuation and stopwords
            if len(cleanedtokens) > 0:
                sentence_embedding = fasttextembed(cleanedtokens)
                print(sentence_embedding.shape)
                if sentence_embedding is not None:
                    x_array.append(sentence_embedding)
                    y_array.append(labelmap[label])
                else:
                    print("sentence embedding is None")
        else:
            text = remove(text)
            tokens = tokenize(text)
            cleanedtokens = [word for word in tokens if word not in allstopwords and word not in string.punctuation]  # removing punctuation and stopwords
            if len(cleanedtokens) > 0:
                sentence_ids = bpembencode(cleanedtokens)
                if sentence_ids is not None:
                    x_array.append(sentence_ids)
                    # print(sentence_ids.shape)
                    y_array.append(labelmap[label])
                else:
                    print("sentence embedding is None")
    np.save(f'../embeddings/{embeddingmodel}_X.npy', np.array(x_array).astype(np.float32))
    np.save(f'../embeddings/{embeddingmodel}_Y.npy', np.array(y_array).astype(np.float32))
    return np.array(x_array), np.array(y_array)