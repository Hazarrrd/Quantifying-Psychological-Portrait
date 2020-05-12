import pandas as pd

# Number of distinct countries in the dataset
N_COUNTIRES=58

# Mapping from labelt to actual question content
QUESTIONS = {
    "EXT1": "I am the life of the party.",
    "EXT2": "I don't talk a lot.",
    "EXT3": "I feel comfortable around people.",
    "EXT4": "I keep in the background.",
    "EXT5": "I start conversations.",
    "EXT6": "I have little to say.",
    "EXT7": "I talk to a lot of different people at parties.",
    "EXT8": "I don't like to draw attention to myself.",
    "EXT9": "I don't mind being the center of attention.",
    "EXT10": "I am quiet around strangers.",
    "EST1": "I get stressed out easily.",
    "EST2": "I am relaxed most of the time.",
    "EST3": "I worry about things.",
    "EST4": "I seldom feel blue.",
    "EST5": "I am easily disturbed.",
    "EST6": "I get upset easily.",
    "EST7": "I change my mood a lot.",
    "EST8": "I have frequent mood swings.",
    "EST9": "I get irritated easily.",
    "EST10": "I often feel blue.",
    "AGR1": "I feel little concern for others.",
    "AGR2": "I am interested in people.",
    "AGR3": "I insult people.",
    "AGR4": "I sympathize with others' feelings.",
    "AGR5": "I am not interested in other people's problems.",
    "AGR6": "I have a soft heart.",
    "AGR7": "I am not really interested in others.",
    "AGR8": "I take time out for others.",
    "AGR9": "I feel others' emotions.",
    "AGR10": "I make people feel at ease.",
    "CSN1": "I am always prepared.",
    "CSN2": "I leave my belongings around.",
    "CSN3": "I pay attention to details.",
    "CSN4": "I make a mess of things.",
    "CSN5": "I get chores done right away.",
    "CSN6": "I often forget to put things back in their proper place.",
    "CSN7": "I like order.",
    "CSN8": "I shirk my duties.",
    "CSN9": "I follow a schedule.",
    "CSN10": "I am exacting in my work.",
    "OPN1": "I have a rich vocabulary.",
    "OPN2": "I have difficulty understanding abstract ideas.",
    "OPN3": "I have a vivid imagination.",
    "OPN4": "I am not interested in abstract ideas.",
    "OPN5": "I have excellent ideas.",
    "OPN6": "I do not have a good imagination.",
    "OPN7": "I am quick to understand things.",
    "OPN8": "I use difficult words.",
    "OPN9": "I spend time reflecting on things.",
    "OPN10": "I am full of ideas."
}

# These are questions which corelate positively to trait 
# according to Big Five model
POSITIVE_QUESTIONS = ['EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9',
                    'EST1', 'EST3', 'EST5', 'EST6', 'EST7', 
                    'EST8', 'EST9', 'EST10',
                    'AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10',
                    'CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10', 
                    'OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 
                    'OPN10']

# These are questions which corelate negatively to trait 
# according to Big Five model
NEGATIVE_QUESTIONS = list(set(QUESTIONS.keys()) - set(POSITIVE_QUESTIONS))

def load_joint(n_samples = 1_000) -> pd.DataFrame:
    """Load and clean dataset. Then randomly return frame consisting of
    n_samples entries

    Keyword Arguments:
        n_samples {int} -- Number of samples from dataset to return (default: {1_000})

    Returns:
        sampled_dataset -- Subsampled dataset with non-intresting columns removed
    """

    dataset = pd.read_csv("dataset/data-final.csv", sep='\t')

    col_times = filter(lambda x: x.find("_E") != -1, list(dataset.columns))
    dataset_clean = dataset.drop(col_times, axis=1)
    dataset_clean = dataset_clean.drop(["screenw", "screenh", "testelapse", "endelapse", "IPC", "lat_appx_lots_of_err", "long_appx_lots_of_err"], axis=1)
    dataset_clean = dataset_clean.drop(["dateload", "introelapse"], axis=1)
    
    # Group unique
    counts = dataset_clean['country'].value_counts().to_dict()
    def group_low_pop(x):
        if pd.isna(x):
            return None

        return x if (counts[x] > 1_000) and (x != "NONE") else "OTHER"
    dataset_clean['country'] = dataset_clean['country'].apply(group_low_pop)

    # Take subsample for faster calculations
    return dataset_clean.sample(n=n_samples, random_state=42)


def load_train_test(n_samples = 2_000) -> (pd.DataFrame, pd.DataFrame):
    """Create a train and tests datasets

    Keyword Arguments:
        n_samples {int} -- Number of samples from dataset to return (default: {1_000})

    Returns:
        (train, test) -- Cleaned dataset split in 75 / 25 ratio into train and test
    """

    dataset = load_joint(n_samples)
    num_train = int(0.75 * len(dataset))
    
    return (dataset.iloc[:num_train, :], dataset.iloc[num_train:, :])


def col_to_text(col: str) -> str:
    """Transforms column name into full question

    Arguments:
        col {str} -- Name of the column in dataframe

    Returns:
        str -- Expansion of the column name into full question
    """
    
    return QUESTIONS[col]

def positive_correlation(col):
    positively_keyed = ['EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9',
                    'EST1', 'EST3', 'EST5', 'EST6', 'EST7', 
                    'EST8', 'EST9', 'EST10',
                    'AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10',
                    'CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10', 
                    'OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 
                    'OPN10']
    
    return col in positively_keyed

def empty_df():
    """Returns empty dataframe with columns exactly the same as in 
    dataset file

    Returns:
        pd.DataFrame -- Dataframe with 0 rows and columns identical to dataset
    """

    empty = {
        key: [] 
        for key in QUESTIONS
    }
    
    empty['country'] = []
    
    return pd.DataFrame(empty)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    Source: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()