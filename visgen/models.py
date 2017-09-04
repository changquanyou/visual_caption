"""
Visual Genome Python API wrapper, models
"""


class Image:
    """
    Image.
      ID         int
      url        hyperlink string
      width      int
      height     int
    """

    def __init__(self, image_id, url, width, height, coco_id, flickr_id):
        self.image_id = image_id
        self.url = url
        self.width = width
        self.height = height
        self.coco_id = coco_id
        self.flickr_id = flickr_id

    def __str__(self):
        return "image_id={}, width={}, height={}, url={}, coco_id={}, flickr_id={}".format(
            self.image_id, self.width, self.height, self.url, self.coco_id, self.flickr_id)


        # return 'id: %d, coco_id: %d, flickr_id: %d, width: %d, url: %s' \
        #        % (self.image_id, -1 if self.coco_id is None else self.coco_id,
        #           -1 if self.flickr_id is None else self.flickr_id, self.width, self.url)

    def __repr__(self):
        return str(self)


class Region:
    """
    Region.
      image 		       int
      phrase           string
      x                int
      y                int
      width            int
      height           int
    """

    def __init__(self, region_id, image_id, phrase, x, y, width, height):
        self.region_id = region_id
        self.image_id = image_id
        self.phrase = phrase
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        return 'regoin_id: %d, x: %d, y: %d, width: %d, height: %d, phrase: %s, image_id: %d' % \
               (self.region_id, self.x, self.y, self.width, self.height, self.phrase, self.image_id)

    def __repr__(self):
        return str(self)


class Graph:
    """
    Graphs contain objects, relationships and attributes
      image            Image
      bboxes           Object array
      relationships    Relationship array
      attributes       Attribute array
    """

    def __init__(self, image, objects, relationships, attributes):
        self.image = image
        self.objects = objects
        self.relationships = relationships
        self.attributes = attributes


class Object:
    """
    Objects.
      id         int
      x          int
      y          int
      width      int
      height     int
      names      string array
      synsets    Synset array
    """

    def __init__(self, object_id, x, y, width, height, names, synsets):
        self.object_id = object_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.names = names
        self.synsets = synsets

    def __str__(self):
        name = self.names[0] if len(self.names) != 0 else 'None'
        return '%s' % (name)

    def __repr__(self):
        return str(self)


class Relationship:
    """
    Relationships. Ex, 'man - jumping over - fire hydrant'.
        subject    int
        predicate  string
        object     int
        rel_canon  Synset
    """

    def __init__(self, relationship_id, subject_id, predicate, object_id, synset):
        self.relationship_id = relationship_id
        self.subject_id = subject_id
        self.predicate = predicate
        self.object_id = object_id
        self.synset = synset

    def __str__(self):
        return "%d: %s %s %s" % (self.relationship_id, self.subject_id, self.predicate, self.object_id)

    def __repr__(self):
        return str(self)


class Attribute:
    """
    Attributes. Ex, 'man - old'.
      subject    Object
      attribute  string
      synset     Synset
    """

    def __init__(self, attribute_id, subject, attribute, synset):
        self.id = attribute_id
        self.subject = subject
        self.attribute = attribute
        self.synset = synset

    def __str__(self):
        return "%d: %s is %s" % (self.attribute_id, self.subject, self.attribute)

    def __repr__(self):
        return str(self)


class QA:
    """
    Question Answer Pairs.
      ID         int
      image      int
      question   string
      answer     string
      q_objects  QAObject array
      a_objects  QAObject array
    """

    def __init__(self, qa_id, image, question, answer, question_objects, answer_objects):
        self.qa_id = qa_id
        self.image = image
        self.question = question
        self.answer = answer
        self.q_objects = question_objects
        self.a_objects = answer_objects

    def __str__(self):
        return 'id: %d, image: %d, question: %s, answer: %s' \
               % (self.qa_id, self.image.id, self.question, self.answer)

    def __repr__(self):
        return str(self)


class QAObject:
    """
    Question Answer Objects are localized in the image and refer to a part
    of the question text or the answer text.
      start_idx          int
      end_idx            int
      name               string
      synset_name        string
      synset_definition  string
    """

    def __init__(self, start_idx, end_idx, name, synset):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.name = name
        self.synset = synset

    def __repr__(self):
        return str(self)


class Synset:
    """
    Wordnet Synsets.
      name       string
      definition string
    """

    def __init__(self, name, definition):
        self.name = name
        self.definition = definition

    def __str__(self):
        return '{} - {}'.format(self.name, self.definition)

    def __repr__(self):
        return str(self)
