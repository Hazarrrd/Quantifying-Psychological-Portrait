from matplotlib import pyplot as plt
import seaborn as sns



def plot_data(df, properties=dict()):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 6),
        sharey=True,
        sharex=True
    )
    positive_category = df[df[properties.get('category', 'category')] == 1]
    negative_category = df[df[properties.get('category', 'category')] == 0]
    sns.scatterplot(negative_category[properties.get('x', 'x')],
                negative_category[properties.get('y', 'y')],
                ax=ax[0])
    cat_name = properties.get('category_labels', dict())
    ax[0].set(xlabel=properties.get('x_label', 'x'),
              ylabel=properties.get('y_label', 'y'),
              title=cat_name.get(0, ''))
    sns.scatterplot(positive_category[properties.get('x', 'x')],
                    positive_category[properties.get('y', 'y')],
                    ax=ax[1])
    ax[1].set(xlabel=properties.get('x_label', 'x'),
              ylabel=properties.get('y_label', 'y'),
              title=cat_name.get(1, ''))

