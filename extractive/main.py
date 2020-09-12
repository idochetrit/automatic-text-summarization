import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import en_core_web_sm

def summarization():

  with open("./stories/d3370f0d60746aebcc5f61a068805b8545357e6f.story", "r", encoding="utf-8") as f:
    text = " ".join(f.readlines())
    core = en_core_web_sm.load()

  doc = core(text)
  # clean sentences
  corpus = [sent.text.lower() for sent in doc.sents]
  STOP_WORDS.add("@highlight")
  cv = CountVectorizer(stop_words=list(STOP_WORDS))
  cv_fit = cv.fit_transform(corpus)
  word_list = cv.get_feature_names()
  count_list = cv_fit.toarray().sum(axis=0)
  
  # zip it in a way that pair word and the its count
  word_frequency = dict(zip(word_list, count_list))
  words_freqs = sorted(word_frequency.values())
  higher_word_frequencies = [word for word,
                             freq in word_frequency.items() if freq in words_freqs[-3:]]
  print("higher frequency words : ", higher_word_frequencies)

  higher_frequency = words_freqs[-1]
  # normalise the frequencies values
  for word in word_frequency.keys():
    word_frequency[word] = (word_frequency[word]/higher_frequency)

  sentence_rank = {}
  for sent in doc.sents:
    for word in sent:
      if word.text.lower() in word_frequency.keys():
        if sent in sentence_rank.keys():
          sentence_rank[sent] += word_frequency[word.text.lower()]
        else:
          sentence_rank[sent] = word_frequency[word.text.lower()]
      else:
        continue

  # fetch top sentences which have the higher top-freq words
  top_sentences = (sorted(sentence_rank.values())[::-1])
  top_sent = top_sentences[:3]

  summary = []
  for sent, strength in sentence_rank.items():
    if strength in top_sent:
      summary.append(sent)

  return text, summary


if __name__ == '__main__':
  text, summary = summarization()
  print("\nOriginal text: \n {} \n".format(text))

  print("\nSummary:")
  for i in summary:
    print(i, end=" ")
