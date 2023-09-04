import glob, os
import re
from tqdm import tqdm
import pickle

def main():
    genre_dict = {}
    with open('genre.txt','w') as f:
        for file in tqdm(glob.glob("labels/*/*.txt")):
            genre = re.search('id_list_(.*).txt',file).group(1).lower()
            IDs = open(file)
            full_IDs = IDs.readlines()
            for ID in full_IDs:
                genre_dict[ID[:-1]] = genre
    
    with open('genre.pickle','wb') as fw:
        pickle.dump(genre_dict,fw)
    print(genre_dict)

if __name__=="__main__":
    main()