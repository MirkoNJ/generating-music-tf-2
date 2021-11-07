import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os

def lineplot(arr, v, l, t, max):
    y1 = np.mean(arr['train_' + v], axis=1)[0:max]
    y2 = np.mean(arr['val_' + v], axis=1)[0:max]
    fig = px.line(y=[y1,y2], title = t)
    fig.data[0].name = "Training"
    fig.data[1].name = "Validation"
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text=l)
    return fig

def plot_confusion_matrix(arr, v, l, t, max):
    perc = arr[v][0:max]  / np.sum(arr[v][0:max] , axis=1, keepdims=True) * 100
    fig = px.line(y=[perc[:,0,0],perc[:,1,0], perc[:,0,1], perc[:,1,1]], title = t + l)
    fig.data[0].name = "Predicted: Not "   + l + ";  True: Not "   + l
    fig.data[1].name = "Predicted:       " + l + ";  True: Not "   + l
    fig.data[2].name = "Predicted: Not "   + l + ";  True:       " + l
    fig.data[3].name = "Predicted:       " + l + ";  True:       " + l
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text='Percentage')
    return fig

def lineplot_matplot(arr, v1, v2, t, max, tb):
    perc_train = arr[v1][0:max]  / np.sum(arr[v1][0:max] , axis=1, keepdims=True) * 100
    perc_val   = arr[v2][0:max]  / np.sum(arr[v2][0:max] , axis=1, keepdims=True) * 100
    labels = ['4', '8', '16', '32', '64']
    locs=[4.0, 8.0, 16.0, 32.0, 64]
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 6), sharey=True, constrained_layout=True)

    y1 = perc_train[:,0,0]
    y2 = perc_train[:,1,1]
    y3 = perc_val[:,0,0]
    y4 = perc_val[:,1,1]
    x = range(1,len(y1)+1)
    

    ax1.plot(x, y1, color='black')
    ax1.plot(x, y2, color='black' , linestyle='dashed')
    ax1.set_title(r'Training')
    ax1.set_ylabel("Percentage")
    ax1.set_xlabel("Epoch")
    ax1.set_xticklabels(labels)
    ax1.set_xticks(locs)    
    ax2.plot(x, y3, color='black', label= "True Not " + t + " \nPredicted Not " + t)
    ax2.plot(x, y4, color='black' , linestyle='dashed', label = "True " + t + " \nPredicted "+ t)
    ax2.set_xlabel("Epoch")
    ax2.set_title(r'Validation')
    ax2.legend(loc=tb+" right")
    ax2.set_xticklabels(labels)
    ax2.set_xticks(locs)    

    fig.suptitle(t)
    plt.rcParams.update({'font.size': 22})
    plt.savefig(os.path.join(t+'.png'),  facecolor="w", transparent= False, dpi=300, format='png') #, bbox_inches='tight', transparent= False
    return plt

def lineplot_alphas(arr, v, l, t, max):
    y1 = np.mean(arr['alpha_1.0_train_'   + v], axis=1)[0:max]
    y2 = np.mean(arr['alpha_1.0_val_'     + v], axis=1)[0:max]
    y3 = np.mean(arr['alpha_0.1_train_'   + v], axis=1)[0:max]
    y4 = np.mean(arr['alpha_0.1_val_'     + v], axis=1)[0:max]
    y5 = np.mean(arr['alpha_0.01_train_'  + v], axis=1)[0:max]
    y6 = np.mean(arr['alpha_0.01_val_'    + v], axis=1)[0:max]
    y7 = np.mean(arr['alpha_0.001_train_' + v], axis=1)[0:max]
    y8 = np.mean(arr['alpha_0.001_val_'   + v], axis=1)[0:max]
    fig = px.line(y=[y1,y2, y3, y4, y5 ,y6 ,y7 ,y8], title = t)
    fig.data[0].name = "Training Alpha 1"
    fig.data[1].name = "Validation Alpha 1"
    fig.data[2].name = "Training Alpha 0.1"
    fig.data[3].name = "Validation Alpha 0.1"
    fig.data[4].name = "Training Alpha 0.01"
    fig.data[5].name = "Validation Alpha 0.01"
    fig.data[6].name = "Training Alpha 0.001"
    fig.data[7].name = "Validation Alpha 0.001"
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text=l)
    return fig

def lineplot_matplot_alphas(arr, v, l, t, max, tb):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0.4,wspace=0.4)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex=True, sharey=True)
    labels = ['1', '2', '4', '8', '16', '32']
    locs=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    y1 = np.mean(arr['alpha_1.0_train_'   + v], axis=1)[0:max]
    y2 = np.mean(arr['alpha_1.0_val_'     + v], axis=1)[0:max]
    y3 = np.mean(arr['alpha_0.1_train_'   + v], axis=1)[0:max]
    y4 = np.mean(arr['alpha_0.1_val_'     + v], axis=1)[0:max]
    y5 = np.mean(arr['alpha_0.01_train_'  + v], axis=1)[0:max]
    y6 = np.mean(arr['alpha_0.01_val_'    + v], axis=1)[0:max]
    y7 = np.mean(arr['alpha_0.001_train_' + v], axis=1)[0:max]
    y8 = np.mean(arr['alpha_0.001_val_'   + v], axis=1)[0:max]
    x = range(1,len(y1)+1)

    ax1.set_ylabel(t)
    # ax2.set_ylabel(t)
    ax3.set_ylabel(t)
    # ax4.set_ylabel(t)
    # ax1.set_xlabel('Epoch')
    # ax2.set_xlabel('Epoch')
    ax3.set_xlabel('Epoch')
    ax4.set_xlabel('Epoch')


    ax1.plot(x, y1, color='black')
    ax1.plot(x, y2, color='black' , linestyle='dashed')
    ax1.set_xticklabels(labels)
    ax1.set_xticks(locs)        
    ax1.set_title(r'$\alpha$ = 1')
    ax2.plot(x, y3, color='black')
    ax2.plot(x, y4, color='black' , linestyle='dashed')
    ax2.set_title(r'$\alpha$ = 0.1')
    ax2.set_xticklabels(labels)
    ax2.set_xticks(locs)    
    ax3.plot(x, y5, color='black')
    ax3.plot(x, y6, color='black' , linestyle='dashed')
    ax3.set_title(r'$\alpha$ = 0.01')
    ax3.set_xticklabels(labels)
    ax3.set_xticks(locs)    
    ax4.plot(x, y7, color='black', label='Training')
    ax4.plot(x, y8, color='black', linestyle='dashed', label='Validation')
    ax4.set_title(r'$\alpha$ = 0.001')
    ax4.legend(loc=tb+" right")
    ax4.set_xticklabels(labels)
    ax4.set_xticks(locs)    
    # fig = px.line(y=[y1,y2, y3, y4, y5 ,y6 ,y7 ,y8], title = t)
    # fig.data[0].name = "Training Alpha 1"
    # fig.data[1].name = "Validation Alpha 1"
    # fig.data[2].name = "Training Alpha 0.1"
    # fig.data[3].name = "Validation Alpha 0.1"
    # fig.data[4].name = "Training Alpha 0.01"
    # fig.data[5].name = "Validation Alpha 0.01"
    # fig.data[6].name = "Training Alpha 0.001"
    # fig.data[7].name = "Validation Alpha 0.001"
    # fig.update_xaxes(title_text='Epoch')
    # fig.update_yaxes(title_text=l)

    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    fig.suptitle(l + ': ' + t)
    plt.rcParams.update({'font.size': 22})
    plt.savefig(os.path.join(l+'.png'),  facecolor="w", transparent= False, dpi=300, format='png') #, bbox_inches='tight', transparent= False
    return fig