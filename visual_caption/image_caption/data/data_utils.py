# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import skimage.color
import skimage.io


class ImageCaptionDataUtils(object):

    def load_embeddings(self):
        pass

    @staticmethod
    def load_image_raw(image_file):
        """
        load image data base on given image id
        :param image_id: image id
        :return: image data
        """
        current_image = skimage.io.imread(image_file)
        if len(current_image.shape) == 2 or current_image.shape[2] == 1:  # this is to convert a gray to RGB image
            current_image = skimage.color.gray2rgb(current_image)  #
        return current_image

    def _pad_sequences(self, sequences, pad_token, max_length):
        """
        args:
            sequences : a generator of list or tuple
            pad_token : the char to pad with
        returns:
            a list of list where each list has same length
        """
        sequence_padded, sequence_lengths = [], []
        for seq in sequences:
            s = list(seq)
            seq_ = seq[:max_length] + [pad_token] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_lengths += [min(len(s), max_length)]
        return sequence_padded, sequence_lengths

    def pad_sequences(self, sequences, pad_token, max_length):
        """
        args :
            sequences : a generator of list or tuple
            pad_token : the char to pad with
            max_length: max length of sequence
        returns :
            a list of list where each sublist has same length
        """
        sequence_padded_list, sequence_length_list = self._pad_sequences(sequences, pad_token, max_length)
        return sequence_padded_list, sequence_length_list

    def convert_id_seqs(self, captions, token2id_dict):
        result = [self.convert_id_seq(caption=caption, token2id_dict=token2id_dict) for caption in captions]
        return result

    def convert_id_seq(self, caption, token2id_dict):
        """
        convert chars in caption into id
        :param caption:
        :param char2id:
        :return:
        """
        caption = caption.replace(' ', '')
        caption = caption.replace('\n', '')
        chars = [char for char in caption]
        chars.insert(0, "<S>")
        chars.append("<\S>")
        result = []
        for idx, char in enumerate(chars):
            if char not in token2id_dict:
                char = "#UNKNOWN#"
            id = token2id_dict[char]
            result.append(id)
        return result
