Project Organization
------------

    │
    ├── data/                 <- raw data extracted from event registry plus transformed data (not on github)
    │
    ├── figures/              <- figures saved by notebooks
    │
    ├── models/               <- serialised models, LDA & classifers (not on github)
    │                         
    ├── notebooks/answers/    <- notebooks with answers
    ├── notebooks/workspace/  <- intermediate notebooks
    │
    ├── phq_utils/            <- python files with utils functions
    │
    ├── README.md             <- the top-level README for people checking this project.



Project Dependencies
------------

After cloning the repo your python (project has been tested with Python 3.6) need to be able to access phq_utils. This can be done by adding a symbolic link:

```bash
ln -s /path/to/my/phq_utils /path/to/anaconda/lib/python3.6/site-packages/
```

The project use various python packages among which: pandas, scikit-learn, spacy, gensim, nltk, keras, lightgbm, seaborn
    


