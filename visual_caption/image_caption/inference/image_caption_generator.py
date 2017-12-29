"""Class for generating captions from an image-to-text model.
   This is based on Google's https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/caption_generator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from visual_caption.image_caption.inference.caption import Caption, TopN


class ImageCaptionGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self, model, vocab, token_start, token_end,
                 beam_size=3, max_caption_length=24,
                 length_normalization_factor=0.0):
        """Initializes the generator.
        Args:
          model: Object encapsulating a trained image-to-text model. Must have
            methods feed_image() and inference_step(). For example, an instance of
            InferenceWrapperBase.
          vocab: A Vocabulary object.
          beam_size: Beam size to use when generating captions.
          max_caption_length: The maximum caption length before stopping the search.
          length_normalization_factor: If != 0, a number x such that captions are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of captions depending on their lengths. For example, if
            x > 0 then longer captions will be favored.
        """
        self.vocab = vocab
        self.model = model

        self.token_start = token_start
        self.token_end = token_end

        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def _initial_state(self, sess, image_feature):
        # get initial state using image feature
        feed_dict = {self.model.image_feature: image_feature}
        fetches = [self.model.language_initial_state]
        language_initial_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        return language_initial_state

    def _inference_step(self, sess, image_feature,
                        input_feed_list, state_feed_list):
        softmax_outputs = []
        new_state_outputs = []
        fetches = [self.model.softmax, self.model.language_new_state]
        for input_feed, state in zip(input_feed_list, state_feed_list):
            state_feed = state
            feed_dict = {
                self.model.input_feed: input_feed,
                self.model.language_state_feed: state_feed,
                self.model.image_feature: image_feature
            }
            softmax, language_new_state = sess.run(
                fetches=fetches, feed_dict=feed_dict)
            softmax_outputs.append(softmax)
            new_state_outputs.append(language_new_state)
        return softmax_outputs, new_state_outputs, None

    def beam_search(self, sess, image_feature):
        """Runs beam search caption generation on a single image.
        Args:
          sess: TensorFlow Session object.
          image_feature: extracted visual feature of one image.
        Returns:
          A list of Caption sorted by descending score.
        """
        token_start_id = self.vocab[self.token_start]
        token_end_id = self.vocab[self.token_end]

        # Feed in the image to get the initial state.
        initial_state = self._initial_state(sess, image_feature)
        initial_beam = Caption(sentence=[token_start_id], state=initial_state[0],
                               logprob=0.0, score=0.0, metadata=[""])
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = [np.array([c.sentence[-1]]) for c in partial_captions_list]
            state_feed = [c.state for c in partial_captions_list]

            softmax, new_states, metadata = self._inference_step(
                sess, image_feature, input_feed, state_feed)
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
