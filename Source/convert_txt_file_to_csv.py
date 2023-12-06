import pandas as pd


def get_feature_columns():
    cols = ["C", "qid"]
    for i in range(700):
        cols.append("X" + str(i))

    return cols


def init_new_row():
    new_row = {"C": 0, "qid": 1}

    for i in range(700):
        new_row["X" + str(i)] = 0.0

    return new_row


def convert_text_file_into_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        df = pd.DataFrame(columns=get_feature_columns())
        for line in file:
            splits = line.split()
            new_row = init_new_row()
            for i in range(len(splits)):
                if i == 0:
                    new_row["C"] = int(splits[i])
                elif i == 1:
                    new_row["qid"] = int(splits[i].split(":")[1])
                else:
                    feature_index = str(splits[i].split(":")[0])
                    feature_value = float(splits[i].split(":")[1])
                    new_row["X" + feature_index] = feature_value

            df.loc[len(df)] = new_row

        return df


if __name__ == "__main__":
    df_train = convert_text_file_into_csv("../Data/set2.train.txt")
    df_train.to_csv("../Data/set2.train.csv", index=False)

    df_test = convert_text_file_into_csv("../Data/set2.test.txt")
    df_test.to_csv("../Data/set2.test.py.csv", index=False)



