import pandas as pd
import torch
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def label_shuffle_data(path="/workspace/ARPIX/ReliableQA/build_dataset/dataset_final.csv"):
    df = pd.read_csv(path)

    # class 정보를 string에서 int로 변환
    le = LabelEncoder()
    result = le.fit_transform(df['class'])
    print(le.classes_)

    df['label'] = pd.Series(result)
    df = shuffle(df)
    df = df.reset_index(drop=True)
    return df


def train_val_test_split(df, val_r, test_r):
    trainval, test = train_test_split(
        df, test_size=test_r, stratify=df['class'])
    # stratify를 사용해 각 split의 data distribution 일치

    test_class_proportions = get_class_proportions(test)

    real_val_r = val_r / (1-test_r)
    train, val = train_test_split(
        trainval, test_size=real_val_r, stratify=trainval['class'])

    train_class_proportions = get_class_proportions(train)
    val_class_proportions = get_class_proportions(val)
    print("train dataset of {} data with a proportion of {}".format(len(train),
     train_class_proportions))
    print("val dataset of {} data with a proportion of {}".format(len(val),
     val_class_proportions))
    print("test dataset of {} data with a proportion of {}".format(len(test),
     test_class_proportions))

    return train, val, test

def get_class_counts(df):  # df에서 각 클래스의 아이템 수
    grp = df.groupby(['class'])['Unnamed: 0'].nunique()
    return ({key: grp[key] for key in list(grp.keys())})


def get_class_proportions(df):  #df에서 각 클래스의 아이템 비율
    class_counts = get_class_counts(df)
    return {val[0]: round(val[1]/df.shape[0], 4) for val in class_counts.items()}


class CompDataset(Dataset):

    def __init__(self, df):
        self.df_data = df



    def __getitem__(self, index):

        # get the sentence from the dataframe
        sentence1 = self.df_data.loc[index, 'text']
        # sentence2 = self.df_data.loc[index, 'hypothesis']


        encoded_dict = tokenizer.encode_plus(
                    sentence1,           # Sentences to encode.
                    add_special_tokens = True,      # Add the special tokens.
                    max_length = MAX_LEN,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',          # Return pytorch tensors.
               )
        
        # These are torch tensors.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        
        # Convert the target to a torch tensor
        target = torch.tensor(self.df_data.loc[index, 'label'])

        sample = (padded_token_list, att_mask, target)

        return sample


    def __len__(self):
        return len(self.df_data)  
    

class TestDataset(Dataset):

    def __init__(self, df):
        self.df_data = df


    def __getitem__(self, index):

        # get the sentence from the dataframe
        sentence1 = self.df_data.loc[index, 'text']
        # sentence2 = self.df_data.loc[index, 'hypothesis']


        encoded_dict = tokenizer.encode_plus(
                    sentence1,           # Sentence to encode.
                    add_special_tokens = True,      # Add the special tokens.
                    max_length = MAX_LEN,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',          # Return pytorch tensors.
               )
        
        # These are torch tensors.
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]                    

        sample = (padded_token_list, att_mask)

        return sample


    def __len__(self):
        return len(self.df_data)
    
df = label_shuffle_data()
train, val, test = train_val_test_split(df, 0.2, 0.2)