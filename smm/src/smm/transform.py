from smm.message import Message


class ModelSecrets


class Transform:
    """
    transform data
    """

    def __init__(self):
        self.translator = self.set_translator()
        self.classifier = self.set_classifier()
        self.wordfreqer = self.set_wordfreqer()
        self.anonymizer = self.set_anonymizer()
        self.geolocator = self.set_geolocator()


    def set_translator(self):
        # TBI
        # default: huggingface, detect language and translate to english

    def set_classifier(self):
        # TBI
        # default: huggingface, classify pos/neg sentiment

    def set_wordfreqer(self):
        # TBI

    def set_anonymizer(self):
        # TBI
        # default: anonymization-app

    def set_geolocator(self):
        # TBI

    def process_data(self,
                     messages,
                     translate=False,
                     classify=False,
                     wordfreq=False,
                     anonymize=False,
                     geolocate=False):
        if translate:
            # TBI translate messages
            messages = []
        if classify:
            # TBI classify messages
            messages = []
        if wordfreq:
            # TBI wordfreq messages
            messages = []
        if geolocate:
            # TBI geolocate messages
            messages = []
        if anonymize:
            # TBI anonymize messages
            messages = []

        return messages