"""Class for generating captions from an image-to-text model.
   This is based on Google's https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/caption_generator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from visual_caption.image_caption.inference.caption import Caption, TopN


class CaptionAttentionGenerator(object):
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

    def _initial_state(self, sess, image_feature, region_features):
        fetches = [
            self.model.attend_initial_state,
            self.model.language_initial_state
        ]
        feed_dict = {
            self.model.image_feature: image_feature,
            self.model.region_features: region_features
        }
        attend_initial_state, language_initial_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        return attend_initial_state, language_initial_state

    def _inference_step(self, sess,
                        input_feed_list, state_feed_list,
                        image_feature, region_features):
        fetches = [
            self.model.softmax,
            self.model.attend_new_state,
            self.model.language_new_state
        ]
        # outputs of current rnn_cell
        softmax_outputs = []
        # new states after update
        new_states = []

        # feed input and state step by step for current sequence
        for input_feed, state in zip(input_feed_list, state_feed_list):
            attend_state, language_state = state
            feed_dict = {
                self.model.input_feed: input_feed,
                self.model.attend_state_feed: attend_state,
                self.model.language_state_feed: language_state,
                self.model.image_feature: image_feature,
                self.model.region_features: region_features
            }
            # run single step for inference
            softmax, new_attend_state, new_language_state = sess.run(
                fetches=fetches, feed_dict=feed_dict)
            softmax_outputs.append(softmax)
            new_states.append((new_attend_state, new_language_state))
        return softmax_outputs, new_states, None

    def beam_search(self, sess, image_feature, region_features):
        """Runs beam search caption generation on a single image.
        Args:
          sess: TensorFlow Session object.
          image_feature: extracted image feature from single image.
          image_feature: extracted region features from single image.
        Returns:
          A beam_size list of captions sorted by descending score .
        """
        token_start_id = self.vocab[self.token_start]
        token_end_id = self.vocab[self.token_end]

        # get initial state for rnn_cell
        attention_initial_state, language_initial_state = self._initial_state(
            sess=sess, image_feature=image_feature, region_features=region_features)

        # construct initial beam captions
        initial_state = (attention_initial_state, language_initial_state)
        initial_beam = Caption(sentence=[token_start_id], state=initial_state,
                               logprob=0.0, score=0.0, metadata=[""])
        # for partial captions
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_beam)
        # for complete captions
        complete_captions = TopN(self.beam_size)

        # Run beam search step by step
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()

            # for last token as the input of next step for each caption
            input_feed = [np.array([c.sentence[-1]]) for c in partial_captions_list]
            state_feed = [c.state for c in partial_captions_list]

            # infer steps for each caption
            softmax, new_states, metadata = self._inference_step(
                sess=sess, input_feed_list=input_feed, state_feed_list=state_feed,
                image_feature=image_feature, region_features=region_features
            )
            # for each beam of top K
            for idx, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[idx][0]
                state = new_states[idx]

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
                        metadata_list = partial_caption.metadata + [metadata[idx]]
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
