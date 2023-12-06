import math
import random


if __name__ == "__main__":
    # opening the original train file
    with open("../Data/set2.train.original.txt", "r") as input_file:
        lines = input_file.readlines()

    # declaring an empty set
    qid_str_set = set()

    # extracting the qids
    for line in lines:
        qid = (line.split()[1]).split(":")[1]
        qid_str_set.add(qid)

    train_length = math.ceil(len(qid_str_set) * 0.8)

    # randomly sampling 70% train queries
    qid_list = list(map(int, list(qid_str_set)))
    train_qid_list = random.sample(qid_list, train_length)
    train_qid_list.sort()

    # getting remaining 30% test.py queries
    test_qid_list = list(set(qid_list) - set(train_qid_list))
    test_qid_list.sort()


    train_documents = []
    test_documents = []

    # writing to train and test.py file
    for line in lines:
        qid = int((line.split()[1]).split(":")[1])
        if qid in test_qid_list:
            test_documents.append(line)
        else:
            train_documents.append(line)

    with open("../Data/set2.train.txt", "w") as output_file:
        output_file.writelines(train_documents)

    with open("../Data/set2.test.txt", "w") as output_file:
        output_file.writelines(test_documents)
