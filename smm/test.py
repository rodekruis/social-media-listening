from smm.message import Message
from smm.transform import Transform
from datetime import datetime
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

messages = []
# messages.append(
#     Message(
#         id_=1,
#         datetime_=datetime.now(),
#         source="Twitter",  # social media source (twitter, telegram..)
#         text="ciao mamma guarda come mi diverto! richiamami al +31648609581",)
# )
messages.append(
    Message(
        id_=2,
        datetime_=datetime.now(),
        source="Twitter",  # social media source (twitter, telegram..)
        text="Hey there I am using WhatsApp, but whatsapp sucks",)
)

my_transformer = Transform()
# my_transformer.set_translator(from_lang="it", to_lang="en", name="HuggingFace", secrets=None)
my_transformer.set_classifier(name="HuggingFace", task="sentiment-analysis")
my_transformer.set_anonymizer(name="anonymization-app")
messages = my_transformer.process_messages(messages, translate=False, classify=True, anonymize=True)

for message in messages:
    print(message.to_dict())

my_transformer.set_wordfreq(wordfreq_lang="en")
df_wordfreq = my_transformer.get_wordfreq(messages)
print(df_wordfreq.head(10))
