from utils import convert

if __name__ == "__main__":
    filename = 'label/admin.jsonl'
    output = "data/train_BIO.txt"
    convert.json2bio(filename,output)