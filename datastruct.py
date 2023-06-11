class paper(object):
    def __init__(self, img, coord):
        self._img = img
        self._coord = coord
        self._lines = []
        self._feature_score = {}
        self._score = 0
        
    def get_img(self):
        return self._img
    
    def get_coord(self):
        return self._coord
        
    def add_line(self, line):
        self._lines.append(line)
    
    def get_lines(self):
        return self._lines
    
    def record_feature_score(self, feature_scores, feature_names):
        self._feature_score = {feature_name : feature_score for feature_name, feature_score in zip (feature_names, feature_scores)}
        
    def get_feature_score(self):
        return self._feature_score
    
    def set_score(self, score):
        self._score = score
    
    def get_score(self):
        return self._score
    
class line(object):
    def __init__(self, img, coord):
        self._img = img
        self._coord = coord
        self._words = []
        self._feature_score = {}
    
    def get_img(self):
        return self._img
    
    def get_coord(self):
        return self._coord
    
    def add_word(self, word):
        self._words.append(word)
        
    def get_words(self):
        return self._words
    
    def record_feature_score(self, feature_scores, feature_names):
        self._feature_score = {feature_name : feature_score for feature_name, feature_score in zip (feature_names, feature_scores)}
        
    def get_feature_score(self):
        return self._feature_score
    
class word(object):
    def __init__(self, img, coord):
        self._img = img
        self._coord = coord
        self._letters = []
        
    def get_img(self):
        return self._img
    
    def get_coord(self):
        return self._coord
    
    def add_letter(self, letter):
        self._letters.append(letter)
        
    def get_letters(self):
        return self._letters
    
class letter(object):
    def __init__(self, img, coord):
        self._img = img
        self._coord = coord
        
    def get_img(self):
        return self._img
    
    def get_coord(self):
        return self._coord

