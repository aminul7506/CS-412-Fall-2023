from Source.utilities import read_data


def convert_clicked_into_binary(input_file_path, output_file_path):
    df = read_data(input_file_path)
    df['C'] = df['C'].apply(lambda x: 1 if x > 2 else 0)
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    convert_clicked_into_binary("../Data/set2.train.csv", "../Data/set2.train.binary.csv")
    convert_clicked_into_binary("../Data/set2.test.csv", "../Data/set2.test.binary.csv")