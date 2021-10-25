import numpy as np
import plotly.express as px

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