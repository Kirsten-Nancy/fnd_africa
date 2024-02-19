from deep_translator import GoogleTranslator

def detect_language_and_translate(text):
    translated_text = GoogleTranslator(source='en', target='sw').translate(text)
    return translated_text

text = 'the vaccine is a gene therapy which is capable of opening the dna up to be editable covid'

print(detect_language_and_translate(text))
# langs_list = GoogleTranslator().get_supported_languages()
# print(langs_list)
