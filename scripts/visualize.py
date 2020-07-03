import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from image_search import create_df_for_map_plot


def plot_input_and_similar(test_img, searches, pd, titles=None):
    '''
    Use this function instead of plot_similar.
    test_img: text path to test image
    searches: tuple of lists. index[0] is a list of similar indices. index[1] is a list of cosine distances (flat_list)
    pd: dataframe of paths
    '''
    idx = searches[0]
    titles = searches[1]

    f = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(left=0.1, bottom=0, right=0.2, top=1.02, wspace=0.02, hspace=0.02)
    plt.axis('Off')
    rows = len(idx)
    cols = 2
    for i, img_idx in enumerate(idx):
        sp = f.add_subplot(rows, cols, 2 * (i + 1))  # want the output pictures on the right side
        sp.axis('Off')
        sp.get_xaxis().set_visible(False)
        sp.get_yaxis().set_visible(False)
        if titles is not None:
            sp.set_title('Cosine sim = %.3f' % titles[i], fontsize=16)
        data = Image.open(pd.iloc[img_idx].path, 'r')
        data = data.convert('RGB')
        data = data.resize((400, 300), Image.ANTIALIAS)
        plt.imshow(data)
        plt.tight_layout()

    # plot test image
    sp = f.add_subplot(rows, cols, rows)  # want the test image in the middle of the column
    sp.axis('Off')
    sp.get_xaxis().set_visible(False)
    sp.get_yaxis().set_visible(False)
    sp.set_title('User Input', fontsize=16)
    data = Image.open(test_img, 'r')
    data = data.convert('RGB')
    data = data.resize((400, 300), Image.ANTIALIAS)
    plt.imshow(data)
    plt.tight_layout()
    plt.autoscale(tight=True)

def plot_map(searches, pd_files):
    '''
    Plots search results on a map
    :param searches: search results from evalTestImage()
    :param pd_files: pandas DataFrame from ImageDataset.get_file_df()
    '''
    df = create_df_for_map_plot(searches, pd_files)
    fig = px.scatter_geo(df, lat='latitude', lon='longitude', text='display')
    fig.show()