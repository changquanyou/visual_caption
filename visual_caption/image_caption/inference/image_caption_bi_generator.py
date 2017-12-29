"""Class for generating captions from an image-to-text model.
   This is based on Google's https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/caption_generator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tf_visgen.mscoco.inference.caption import TopN, Caption


class ImageCaptionBiGenerator(object):
    """
    Class to generate captions from an image-to-text model.

        generate caption_text for single image
        based on: image_feature and region_features
    """

    def __init__(self,
                 model,
                 token_start,
                 token_end,
                 vocab,
                 beam_size=3,
                 max_caption_length=24,
                 length_normalization_factor=0.0):
        """Initializes the generator.
        Args:
          model: Object encapsulating a trained image-to-text model.
          Must have methods
                    feed_image() and
                    inference_step().
            For example, an instance of InferenceWrapperBase.
          vocab: A Vocabulary object.

          beam_size: Beam size to use when generating captions.

          max_caption_length: The maximum caption length before stopping the search.

          length_normalization_factor:
            If != 0, a number x such that captions are scored by logprob/length^x, rather than logprob.
            This changes the relative scores of captions depending on their lengths.
            For example, if x > 0 then longer captions will be favored.
        """
        self.vocab = vocab
        self.model = model
        self.beam_size = beam_size
        self.token_start = token_start
        self.token_end = token_end
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    # get forward and backward initial states for attention model
    def _get_attend_initial_state(self, sess, image_feature):
        feed_dict = {self.model.image_feature: image_feature}
        fetches = [self.model.attend_fw_initial_state, self.model.attend_bw_initial_state]
        attend_fw_initial_state, attend_bw_initial_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        return attend_fw_initial_state, attend_bw_initial_state

    # get forward and backward initial states for language model
    def _get_language_initial_state(self, sess,
                                    attend_fw_initial_state,
                                    attend_bw_initial_state,
                                    image_feature):
        # get language initial states using image feature and region features
        token_start_id = self.vocab[self.token_start]
        token_end_id = self.vocab[self.token_end]
        feed_dict = {
            self.model.input_fw_feed: [token_start_id],
            self.model.input_bw_feed: [token_end_id],
            self.model.attend_fw_state_feed: attend_fw_initial_state,
            self.model.attend_bw_state_feed: attend_bw_initial_state,
            self.model.image_feature: image_feature,
        }
        fetches = [self.model.language_fw_initial_state,
                   self.model.language_bw_initial_state]
        language_fw_initial_state, language_bw_initial_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)

        return language_fw_initial_state, language_bw_initial_state

    def _inference_step(self, sess,
                        input_feed_list, state_feed_list,
                        max_caption_length):
        mask = np.zeros((1, max_caption_length))
        mask[:, 0] = 1
        softmax_outputs = []
        new_states = []

        fetches = [
            self.model.softmax,
            self.model.attend_fw_new_state,
            self.model.attend_bw_new_state,
            self.model.language_fw_new_state,
            self.model.language_bw_new_state
        ]

        for inputs, states in zip(input_feed_list, state_feed_list):
            attend_fw_state_feed, attend_bw_state_feed, \
            language_fw_state_feed, language_bw_state_feed = states

            input_fw, input_bw = inputs
            feed_dict = {
                self.model.input_fw_seqs: input_fw,
                self.model.input_bw_seqs: input_bw,

                self.model.attend_fw_state_feed: attend_fw_state_feed,
                self.model.attend_bw_state_feed: attend_bw_state_feed,

                self.model.language_fw_state_feed: language_fw_state_feed,
                self.model.language_bw_state_feed: language_bw_state_feed,
            }

            # run single step for inference
            softmax, \
            new_attend_fw_state, new_attend_bw_state, \
            new_language_fw_state, new_language_bw_state \
                = sess.run(fetches=fetches,
                           feed_dict=feed_dict)

            softmax_outputs.append(softmax)
            # new_state = (new_state_fw, new_state_bw)
            new_states.append([new_attend_fw_state, new_attend_bw_state, new_language_fw_state, new_language_bw_state])
        return softmax_outputs, new_states, None

    def _inference_fw_step(self, sess,
                           input_fw_feed_list,
                           state_fw_feed_list,
                           image_feature,
                           max_caption_length):
        """
        inference forward one step
        :param sess:
        :param input_fw_feed_list:
        :param state_fw_feed_list:
        :param image_feature:
        :param region_features:
        :param max_caption_length:
        :return:


        """
        mask = np.zeros((1, max_caption_length))
        mask[:, 0] = 1
        fw_softmax_outputs = []
        new_states = []
        fetches = [
            self.model.fw_softmax,
            self.model.attend_fw_new_state,
            self.model.language_fw_new_state,
        ]
        for fw_inputs, fw_states in zip(input_fw_feed_list, state_fw_feed_list):
            # print("fw_inputs={}".format(fw_inputs))
            attend_fw_state, language_fw_state = fw_states
            feed_dict = {
                self.model.input_fw_feed: fw_inputs[0],
                self.model.attend_fw_state_feed: attend_fw_state,
                self.model.language_fw_state_feed: language_fw_state,

                self.model.image_feature: image_feature,
                self.model.region_features: region_features,

                self.model.input_mask: mask,
                self.model.keep_prob: 1.0}
            # run single step for inference
            fw_softmax, new_attend_fw_state, new_language_fw_state = sess.run(
                fetches=fetches, feed_dict=feed_dict)
            fw_softmax_outputs.append(fw_softmax)
            # new_state = (new_state_fw, new_state_bw)
            new_states.append([new_attend_fw_state, new_language_fw_state])
        return fw_softmax_outputs, new_states, None

    def _inference_bw_step(self, sess,
                           input_bw_feed_list,
                           state_bw_feed_list,
                           image_feature,
                           region_features,
                           max_caption_length):
        """
        inference forward one step
        :param sess:
        :param input_bw_feed_list:
        :param state_bw_feed_list:
        :param image_feature:
        :param region_features:
        :param max_caption_length:
        :return:


        """
        mask = np.zeros((1, max_caption_length))
        mask[:, 0] = 1
        bw_softmax_outputs = []
        new_states = []
        fetches = [
            self.model.bw_softmax,
            self.model.attend_bw_new_state,
            self.model.language_bw_new_state,
        ]
        for bw_inputs, bw_states in zip(input_bw_feed_list, state_bw_feed_list):
            # print("fw_inputs={}".format(fw_inputs))
            attend_bw_state, language_bw_state = bw_states
            feed_dict = {

                self.model.input_bw_feed: bw_inputs[0],
                self.model.attend_bw_state_feed: attend_bw_state,
                self.model.language_bw_state_feed: language_bw_state,

                self.model.image_feature: image_feature,
                self.model.region_features: region_features,

                self.model.input_mask: mask,
                self.model.keep_prob: 1.0}
            # run single step for inference
            bw_softmax, new_attend_bw_state, new_language_bw_state = sess.run(
                fetches=fetches, feed_dict=feed_dict)
            bw_softmax_outputs.append(bw_softmax)
            # new_state = (new_state_fw, new_state_bw)
            new_states.append([new_attend_bw_state, new_language_bw_state])
        return bw_softmax_outputs, new_states, None

    def beam_search(self, sess, image_feature, region_features):
        """Runs beam search caption generation on a single image forward and backward.
        Args:
          sess: TensorFlow Session object.
          image_feature: extracted image feature for single image.
          region_features: extracted region features for regions detected from this single image
        Returns:
          A beam_size list of captions sorted by descending score .
        """
        token_start_id = self.vocab[self.token_start]
        token_end_id = self.vocab[self.token_end]

        # Feed in the image to get the initial states.
        # initial states is :
        attend_fw_initial_state, attend_bw_initial_state = \
            self._get_attend_initial_state(sess=sess,
                                           image_feature=image_feature)

        language_fw_initial_state, language_bw_initial_state = \
            self._get_language_initial_state(
                sess=sess,
                attend_fw_initial_state=attend_fw_initial_state,
                attend_bw_initial_state=attend_bw_initial_state,
                image_feature=image_feature)

        initial_fw_states = (attend_fw_initial_state,
                             language_fw_initial_state)

        initial_bw_states = (attend_bw_initial_state,
                             language_bw_initial_state)

        # inference based on language_fw_initial_state
        initial_fw_beam = Caption(sentence=[token_start_id],
                                  state=initial_fw_states,
                                  logprob=0.0,
                                  score=0.0,
                                  metadata=[""])

        # inference based on language_bw_initial_state
        initial_bw_beam = Caption(sentence=[token_end_id],
                                  state=initial_bw_states,
                                  logprob=0.0,
                                  score=0.0,
                                  metadata=[""])

        # for partial captions
        fw_partial_captions = TopN(self.beam_size)
        fw_partial_captions.push(initial_fw_beam)
        # for complete captions
        fw_complete_captions = TopN(self.beam_size)

        # Run beam search step by step forward
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = fw_partial_captions.extract()
            fw_partial_captions.reset()
            input_feed = [np.array([c.sentence[-1]]).reshape(1, 1) for c in partial_captions_list]
            state_feed = [c.state for c in partial_captions_list]
            # infer one step forward
            softmax, new_states, metadata = self._inference_fw_step(
                sess=sess,
                input_fw_feed_list=input_feed,
                state_fw_feed_list=state_feed,
                image_feature=image_feature,
                max_caption_length=self.max_caption_length
            )
            # for each beam of top K
            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i][0]
                state = new_states[i]

                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]

                # Each next word gives a new partial caption.
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    if metadata:
                        metadata_list = partial_caption.metadata + [metadata[i]]
                    else:
                        metadata_list = None
                    if w == token_end_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        fw_complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        fw_partial_captions.push(beam)
            if fw_partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete captions.
        if not fw_complete_captions.size():
            fw_complete_captions = fw_partial_captions

        return fw_complete_captions.extract(sort=True)


class ImageCaptionBackwardGenerator(object):

    def __init__(self, model, token_start, token_end,
                 vocab, beam_size=3, max_caption_length=24,
                 length_normalization_factor=0.0):
        """Initializes the generator.
        Args:
          model: Object encapsulating a trained image-to-text model.
            Must have methods :  initial() and inference_step().
          vocab: A Vocabulary object.
          beam_size: Beam size to use when generating captions.
          max_caption_length: The maximum caption length before stopping the search.
          length_normalization_factor:
            If != 0, a number x such that captions are scored by logprob/length^x, rather than logprob.
            This changes the relative scores of captions depending on their lengths.
            For example, if x > 0 then longer captions will be favored.
        """
        self.vocab = vocab
        self.model = model
        self.beam_size = beam_size
        self.token_start = token_start
        self.token_end = token_end
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    # get forward and backward initial states
    def _initial(self, sess, image_feature):
        # zero state as the initial state
        fetches = [self.model.language_fw_initial_state,
                   self.model.language_bw_initial_state]
        feed_dict = {self.model.image_feature: image_feature}
        language_fw_initial_state, language_bw_initial_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        return language_fw_initial_state, language_bw_initial_state

    def _inference_step(self, sess, input_feed, state_feed, image_feature):
        fetches = [self.model.bw_softmax,
                   self.model.language_bw_new_state]
        feed_dict = {
            self.model.input_bw_feed: input_feed,
            self.model.language_bw_state_feed: state_feed,
            self.model.image_feature: image_feature
        }
        softmax_output, new_language_bw_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        return softmax_output, new_language_bw_state, None

    def beam_search(self, sess, image_feature):
        """Runs beam search caption generation on a single image backward.
        Returns:
          A beam_size list of captions sorted by descending score.
        """
        token_start_id = self.vocab[self.token_start]
        token_end_id = self.vocab[self.token_end]

        # Feed in the image to get the initial states.
        language_fw_initial_state, language_bw_initial_state = self._initial(
            sess=sess, image_feature=image_feature)

        initial_bw_beam = Caption(sentence=[token_end_id],
                                  state=language_bw_initial_state[0],
                                  logprob=0.0, score=0.0, metadata=[""])
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_bw_beam)
        complete_captions = TopN(self.beam_size)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()

            # for idx, partial_caption in enumerate(partial_captions_list):
            #     partial_caption.sentence = list(reversed(partial_caption.sentence))
            #     partial_caption.state = list(reversed(partial_caption.state))

            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])
            softmax, new_states, metadata = self._inference_step(
                sess, input_feed, state_feed,
                image_feature=image_feature)

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i]
                state = new_states[i]
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]
                # Each next word gives a new partial caption.
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    if metadata:
                        metadata_list = partial_caption.metadata + [metadata[i]]
                    else:
                        metadata_list = None
                    if w == token_start_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete captions.
        if not complete_captions.size():
            complete_captions = partial_captions
        return complete_captions.extract(sort=True)

    pass


class ImageCaptionForwardGenerator(object):

    def __init__(self, model, token_start, token_end,
                 vocab, beam_size=3, max_caption_length=24,
                 length_normalization_factor=0.0):
        """Initializes the generator.
        Args:
          model: Object encapsulating a trained image-to-text model.
            Must have methods :  initial() and inference_step().
          vocab: A Vocabulary object.
          beam_size: Beam size to use when generating captions.
          max_caption_length: The maximum caption length before stopping the search.
          length_normalization_factor:
            If != 0, a number x such that captions are scored by logprob/length^x, rather than logprob.
            This changes the relative scores of captions depending on their lengths.
            For example, if x > 0 then longer captions will be favored.
        """
        self.vocab = vocab
        self.model = model
        self.beam_size = beam_size
        self.token_start = token_start
        self.token_end = token_end
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    # get forward and backward initial states
    def _initial(self, sess, image_feature):
        fetches = [self.model.language_fw_initial_state,
                   self.model.language_bw_initial_state]
        feed_dict = {self.model.image_feature: image_feature}
        language_fw_initial_state, language_bw_initial_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        return language_fw_initial_state, language_bw_initial_state

    def _inference_step(self, sess, input_feed, state_feed, image_feature):
        fetches = [self.model.fw_softmax, self.model.language_fw_new_state]
        feed_dict = {
            self.model.input_fw_feed: input_feed,
            self.model.language_fw_state_feed: state_feed,
            self.model.image_feature: image_feature
        }
        softmax_output, new_language_fw_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        return softmax_output, new_language_fw_state, None

    def beam_search(self, sess, image_feature):
        """Runs beam search caption generation on a single image backward.
        Returns:
          A beam_size list of captions sorted by descending score.
        """
        token_start_id = self.vocab[self.token_start]
        token_end_id = self.vocab[self.token_end]
        # Feed in the image to get the initial states.
        language_fw_initial_state, _ = self._initial(
            sess=sess, image_feature=image_feature)

        initial_beam = Caption(
            sentence=[token_start_id],
            state=language_fw_initial_state[0],
            logprob=0.0, score=0.0, metadata=[""])
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])

            softmax, new_states, metadata = self._inference_step(
                sess, input_feed, state_feed, image_feature=image_feature)

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i]
                state = new_states[i]
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]
                # Each next word gives a new partial caption.
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    if metadata:
                        metadata_list = partial_caption.metadata + [metadata[i]]
                    else:
                        metadata_list = None
                    if w == token_end_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete captions.
        if not complete_captions.size():
            complete_captions = partial_captions

        return complete_captions.extract(sort=True)

    pass
