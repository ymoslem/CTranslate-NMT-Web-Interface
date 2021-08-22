# CTranslate-NMT-Web-Interface
Machine Translation (MT) Web Interface for OpenNMT and FairSeq models using *CTranslate* and *Streamlit*.


## Install requirements

It is recommended you first create a virtual environment, and then run:

```
pip3 install -r requirements.txt
```

## Example 1: Upper My Text

With [test.py](test.py), you can run a very simple Streamlit example using the command:

```
streamlit run test.py
```

![streamlit-test](/img/streamlit-test.png)


## Example 2: Translate

With [translate.py](translate.py), you can run a fast web translator using CTranslate2. For this example to work, you have to change the paths to your models; search the code for [Modify] and adjust the following lines.

![streamlit-translate](/img/streamlit-translate.png)

Note: This example assumes SentencePiece was used to prepare the data. If you did not use SentencePiece, remove the relevant lines.


## Example 3: Translate - Multiple

If you want your web interface to support multiple languages and/or be able to translate multiple lines like this, you can refer to [translate-multi.py](advanced/translate-multi.py)

![streamlit-translate-multi](/img/streamlit-translate-multi.png)


## Tutorial

Check the detailed instructions at in this [blog tutorial](https://blog.machinetranslation.io/nmt-web-interface/).
