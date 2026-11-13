# NLP Preprocessing, Named Entity Recognition, and Scaled Dot-Product Attention

A practical neural-network assignment exploring **basic natural language processing (NLP)**, **named entity recognition (NER) with spaCy**, and **scaled dot-product attention**.

The notebook first processes a sentence through tokenization, stopword removal, and stemming. It then uses a pretrained spaCy English model to identify entities and their character positions. Finally, it implements the scaled dot-product attention calculation with NumPy to produce attention weights and a weighted output matrix.

## What This Project Covers

### 1. NLP Preprocessing Pipeline

The first task implements a **basic NLP preprocessing pipeline** using regular expressions and NLTK.

The program:

- tokenizes the input sentence into words and punctuation;
- removes common English stopwords and punctuation;
- uses a **set** for efficient, case-insensitive stopword membership checks;
- applies **Porter stemming** to the remaining words; and
- prints the three stages in order.

**Input sentence:**

> NLP techniques are used in virtual assistants like Alexa and Siri.

The completed pipeline produced the following results:

![NLP preprocessing output](images/1%20NLP%20Preprocessing%20Pipeline.png)


The original tokens include the final period, while the filtered output removes punctuation and stopwords such as `are`, `in`, and `and`. Porter stemming changes `techniques` to `techniqu`, `assistants` to `assist`, and `used` to `use`. The output retains `used` after stopword filtering because it is not in the NLTK English stopword list used by the notebook.

The English stopword set and stemmer are initialized **once**, outside the preprocessing function, so they can be reused without repeated setup.

---

### 2. Named Entity Recognition with spaCy

The second task uses **spaCy's pretrained `en_core_web_sm` model** to extract named entities from a provided sentence.

The program:

- loads the pretrained English model once;
- processes the sentence through spaCy's NLP pipeline;
- identifies entities using `doc.ents`; and
- prints the **entity text**, **entity label**, and **start/end character positions**.

**Input sentence:**

>  LeBron James as the NBA player of the United States in Laker and won the NBA Final MVP in 2020.

The leading space in the actual input string is preserved because character offsets are measured from the original string, including that space.

The saved notebook output is shown below:

![spaCy named entity recognition output](images/2%20Named%20Entity%20Recognition%20with%20SpaCy.png)

| Detected Text | Entity Label | Start | End |
| --- | --- | ---: | ---: |
| LeBron James | PERSON | 1 | 13 |
| NBA | ORG | 21 | 24 |
| the United States | GPE | 35 | 52 |
| Laker | GPE | 56 | 61 |
| MVP | ORG | 84 | 87 |
| 2020 | DATE | 91 | 95 |

`PERSON` identifies a person, `ORG` an organization, `GPE` a geopolitical entity, and `DATE` a date. The start offset is **inclusive** and the end offset is **exclusive**, following spaCy's character-offset convention.

**Important:** These are the model's actual predictions as saved in the submitted notebook, not manually corrected labels. For example, the prediction `Laker → GPE` may be incorrect; NER models can misclassify ambiguous text.

---

### 3. Scaled Dot-Product Attention

The third task implements **scaled dot-product attention from scratch using NumPy** rather than a pretrained attention layer.

The program:

- computes Query–Key scores using `Q @ K.T`;
- scales the scores by the square root of the Key dimension;
- applies a numerically stable **row-wise Softmax** to obtain attention weights;
- multiplies attention weights by the Value matrix to obtain the output; and
- checks the output shapes, normalized weight sums, and expected attention values.

**Formula:**

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The input matrices provided by the assignment are:

```python
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
```

The attention calculation produced **Attention weights and Final output matrix (rounded to three decimals):**

![Scaled dot-product attention weights and final output](images/3%20Scaled%20Dot-Product%20Attention.png)

The attention matrix is **2 × 2** because there are two Query vectors and two Key vectors. Each row sums to approximately 1 after rounding. The final output matrix is **2 × 4** because there are two Query vectors and each Value vector contains four elements.

Query 1 assigns more weight to Value 1, while Query 2 assigns more weight to Value 2. The implementation performs matrix calculations using vectorized NumPy operations rather than explicit loops over Query–Key pairs.

---

## Short-Answer Concepts

Each programming task also includes two short-answer questions, answered in the notebook.

| Topic | Main Point |
| --- | --- |
| Stemming vs. lemmatization | Stemming applies affix-removal rules; lemmatization uses vocabulary and grammar to identify a dictionary form. Both can turn `running` into `run` in a verb context. |
| Stopword removal | Can reduce features for keyword-based tasks, but removing important words such as `not` can reverse sentiment. |
| NER vs. POS tagging | NER identifies named entities and their categories; POS tagging assigns grammatical roles such as noun and verb. |
| NER applications | Healthcare information extraction and legal document analysis can use entity detection to organize unstructured text. |
| Attention scaling | Dividing by `√d_k` helps control large dot-product scores, reducing Softmax saturation and helping training stability. |
| Contextual relationships | Self-attention lets a word gather information from other relevant words, such as using `river` to interpret `bank`. |

---

## Running the Notebook

The project uses **Python in Jupyter Notebook or JupyterLab**.

### Required Libraries

- NumPy
- NLTK
- spaCy
- Jupyter Notebook or JupyterLab
- spaCy English pretrained model: `en_core_web_sm`

Install Python packages from a terminal or command prompt:

```bash
python -m pip install numpy nltk spacy jupyter
```

Download the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Or, from a Jupyter notebook cell, install the packages into the active kernel:

```python
%pip install numpy nltk spacy
```

Then download the spaCy model in a notebook cell:

```python
import sys
!{sys.executable} -m spacy download en_core_web_sm
```

Restart the Jupyter kernel after installation if necessary, then open **`Neural Networks_HA4.ipynb`** and run cells from top to bottom.

### NLTK Stopwords

The notebook checks for the NLTK English stopwords corpus and downloads it on the first run if it is missing. That initial download requires internet access; subsequent runs can reuse the downloaded corpus.

### spaCy Model

The NER task calls:

```python
nlp = spacy.load("en_core_web_sm")
```

The `en_core_web_sm` model **must be installed separately** from the `spacy` Python package. Exact NER predictions may differ across model versions.

---

## Key Takeaways

- NLP preprocessing converts raw text into tokens, removes selected common words, and reduces remaining words to stems.
- Removing stopwords is task-dependent; some seemingly common words carry important meaning.
- spaCy can extract entity text, categories, and precise character offsets from natural-language input.
- NER predictions are not always correct, particularly for ambiguous or awkwardly phrased sentences.
- Scaled dot-product attention measures relevance using Queries and Keys and combines information from Values.
- The Softmax attention weights sum to 1 for each Query row, and attention output dimensions are determined by the number of Queries and the Value dimension.
- Reusing preprocessing objects and using vectorized numerical operations can make code simpler and avoid unnecessary work.

---

## Repository Structure

```text
Neural Networks_HA4/
│
├── Neural Networks_HA4.ipynb
├── README.md
│
└── images/
    ├── 1 NLP Preprocessing Pipeline.png
    ├── 2 Named Entity Recognition with SpaCy.png
    └── 3 Scaled Dot-Product Attention.png
```

---

## Technologies Used

**Python · NumPy · NLTK · spaCy · PorterStemmer · Regex · Jupyter Notebook**
