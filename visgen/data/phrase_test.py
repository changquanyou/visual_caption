def load_phrase_data(phrase):
    data_list = []
    words = str.split(phrase)
    words.insert(0, "#BEGIN#")
    words.append("#END#")
    for word_index, word in enumerate(words):
        if word_index > 0:
            input_seq = words[:word_index]
            target_seq = words[:word_index + 1]
            data_list.append((input_seq, target_seq))
    return data_list


def load_next_word(phrase):
    data_list = []
    words = str.split(phrase)
    words.insert(0, "#BEGIN#")
    words.append("#END#")
    for word_index, word in enumerate(words):
        if 0 < word_index:
            input_seq = words[:word_index]
            target_word = words[word_index]
            data_list.append((input_seq, target_word))
    return data_list


def main():
    phrase = "This is a test sequence !"
    seq_list = load_next_word(phrase)
    for input_seq, target_seq in seq_list:
        print("input_sequence = {}".format(input_seq))
        print("target_sequence = {}".format(target_seq))
        print()


if __name__ == '__main__':
    main()
